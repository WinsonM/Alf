import AVFoundation
import Observation

/// 手表上的录音机：点了开始就一直录，一分钟封一段文件，交给 Courier 传走。
///
/// 为什么一分钟一段：整天录成一个文件的话，手表半路没电就全没了，手机也得等到最后才拿得到。
/// 一分钟一段，录一段传一段，丢最多丢一分钟。段和段之间不停麦——同一个 AVAudioEngine 一直开着，只是换个文件写，
/// 所以两段首尾相接、一句话不会被切掉半个字；下一段的开始时间按上一段写了多少个采样点来算，不看钟。
///
/// 抬腕放下之后还能接着录，靠的是 Info.plist 里的后台音频模式加上一个开着的录音会话。
@Observable
final class Recorder {
    enum Phase: Equatable {
        case idle
        case recording
        case interrupted          // 被电话、Siri 打断了；打断结束会自己接着录
        case failed(String)
    }

    private(set) var phase: Phase = .idle
    private(set) var level: Float = 0            // 眼下的音量，0…1，给界面上的那圈跳动用
    private(set) var since: Date?                // 这一次是什么时候开始录的

    /// 封好一段之后调：文件在哪、这一段是什么时候开始录的。
    var onChunk: ((URL, Date) -> Void)?

    static let spool = URL.appFolder("spool")
    private static let chunkSeconds = 60.0
    private static let spoolLimit: Int64 = 250 * 1024 * 1024      // 攒了这么多还没传走（大约十几个小时），先停下：再录就要把手表塞满了

    private let engine = AVAudioEngine()
    private let lock = NSLock()                  // 写文件在音频的线程上，开始/停止在主线程上
    private var running = false
    private var file: AVAudioFile?
    private var fileURL: URL?
    private var fileStart = Date()
    private var frames: AVAudioFramePosition = 0
    private var sampleRate = 16_000.0
    private var nextStart: Date?
    private var lastLevelAt = Date.distantPast
    private var observers: [NSObjectProtocol] = []

    init() {
        let center = NotificationCenter.default
        observers.append(center.addObserver(forName: AVAudioSession.interruptionNotification, object: nil, queue: .main) { [weak self] note in
            guard let self, let raw = note.userInfo?[AVAudioSessionInterruptionTypeKey] as? UInt,
                  let type = AVAudioSession.InterruptionType(rawValue: raw) else { return }
            if type == .began {
                if self.phase == .recording { self.halt(); self.phase = .interrupted }
            } else if self.phase == .interrupted {
                self.start()                     // 打断结束：接着录。系统没说"可以恢复"也试一次，录不起来会如实显示
            }
        })
        // 换了麦克风（连上、断开耳机）：引擎会自己停，重新起一遍
        observers.append(center.addObserver(forName: .AVAudioEngineConfigurationChange, object: engine, queue: .main) { [weak self] _ in
            guard let self, self.phase == .recording else { return }
            self.halt()
            self.start()
        })
    }

    deinit { observers.forEach(NotificationCenter.default.removeObserver) }

    func toggle() {
        if phase == .recording || phase == .interrupted { stop() } else { start() }
    }

    func start() {
        guard Self.spoolSize() < Self.spoolLimit else {
            phase = .failed("攒着没传走的录音太多了，等手机把它们取走再录")
            return
        }
        let session = AVAudioSession.sharedInstance()
        do {
            try session.setCategory(.record, mode: .default, options: [])
            try session.setActive(true)                      // 采样率手表上不让挑，系统给多少用多少；Mac 那头听之前会统一转成 16 kHz
            let input = engine.inputNode
            let format = input.outputFormat(forBus: 0)
            guard format.sampleRate > 0, format.channelCount > 0 else {
                phase = .failed("麦克风用不了（没给权限？）")
                return
            }
            sampleRate = format.sampleRate
            input.removeTap(onBus: 0)
            input.installTap(onBus: 0, bufferSize: 4096, format: format) { [weak self] buffer, _ in self?.take(buffer) }
            lock.lock(); running = true; lock.unlock()
            engine.prepare()
            try engine.start()
            if since == nil { since = Date() }
            phase = .recording
        } catch {
            halt()
            phase = .failed("录不起来：\(error.localizedDescription)")
        }
    }

    func stop() {
        halt()
        since = nil
        phase = .idle
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
    }

    /// 停麦、把手头这一段封好。不改"现在是什么状态"，由调用的人定。
    private func halt() {
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        lock.lock()
        running = false
        seal()
        nextStart = nil
        lock.unlock()
        level = 0
    }

    /// 音频线程上，每来一小块声音调一次：写进当前这一段，写满一分钟就封口换下一段。
    private func take(_ buffer: AVAudioPCMBuffer) {
        lock.lock()
        defer { lock.unlock() }
        guard running else { return }
        do {
            if file == nil { try open(buffer.format) }
            try file?.write(from: buffer)
            frames += AVAudioFramePosition(buffer.frameLength)
            if Double(frames) >= Self.chunkSeconds * sampleRate { seal() }
        } catch {
            seal()
            DispatchQueue.main.async { [weak self] in
                self?.halt()
                self?.phase = .failed("写不进去：\(error.localizedDescription)")
            }
            return
        }
        // 音量一秒更新五次就够了：再勤只是白白叫醒屏幕
        if Date().timeIntervalSince(lastLevelAt) > 0.2, let data = buffer.floatChannelData?[0], buffer.frameLength > 0 {
            lastLevelAt = Date()
            var sum: Float = 0
            for i in 0..<Int(buffer.frameLength) { sum += data[i] * data[i] }
            let rms = (sum / Float(buffer.frameLength)).squareRoot()
            let shown = min(1, max(0, (20 * log10(max(rms, 1e-6)) + 55) / 45))      // -55 dB 到 -10 dB 摊成 0…1
            DispatchQueue.main.async { [weak self] in self?.level = shown }
        }
    }

    private func open(_ format: AVAudioFormat) throws {
        let start = nextStart ?? Date()
        let url = Self.spool.appendingPathComponent("\(Int64(start.timeIntervalSince1970 * 1000)).m4a")
        let settings: [String: Any] = [
            AVFormatIDKey: kAudioFormatMPEG4AAC,
            AVSampleRateKey: format.sampleRate,
            AVNumberOfChannelsKey: format.channelCount,
            AVEncoderBitRateKey: format.sampleRate > 24_000 ? 32_000 : 24_000,
        ]
        file = try AVAudioFile(forWriting: url, settings: settings, commonFormat: format.commonFormat, interleaved: format.isInterleaved)
        fileURL = url
        fileStart = start
        frames = 0
    }

    /// 封口：关文件、记下下一段从哪一刻接着算、把这一段交出去。不到一秒的零头不要。
    private func seal() {
        guard let url = fileURL else { return }
        let start = fileStart
        let seconds = Double(frames) / sampleRate
        file = nil                               // 放掉它，文件才写完整
        fileURL = nil
        nextStart = start.addingTimeInterval(seconds)
        if seconds >= 1 {
            DispatchQueue.main.async { [weak self] in self?.onChunk?(url, start) }
        } else {
            try? FileManager.default.removeItem(at: url)
        }
    }

    static func spoolSize() -> Int64 {
        let files = (try? FileManager.default.contentsOfDirectory(at: spool, includingPropertiesForKeys: [.fileSizeKey])) ?? []
        return files.reduce(0) { $0 + Int64((try? $1.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0) }
    }
}
