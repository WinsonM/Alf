import Foundation
import Observation
import WatchConnectivity

/// 把封好的录音交给手机。用的是系统的文件传输队列：手机不在身边也能先排着，连上了系统自己送，
/// App 被关掉也不丢。手机确认收到之后才删手表上的这一份。
@Observable
final class Courier: NSObject, WCSessionDelegate {
    private(set) var waiting = 0                 // 还没送到手机的有几段

    func activate() {
        guard WCSession.isSupported() else { return }
        WCSession.default.delegate = self
        WCSession.default.activate()
        refresh()
    }

    func send(_ url: URL, startedAt: Date) {
        guard WCSession.default.activationState == .activated else { refresh(); return }    // 还没连好：先留在手表上，连好之后一起补送
        WCSession.default.transferFile(url, metadata: [Wire.startedAt: startedAt.timeIntervalSince1970, Wire.device: "watch"])
        refresh()
    }

    /// 手表上还留着、却不在传输队列里的录音（上次没送成，或者录的时候还没连好）：重新排上。
    /// 文件名就是开始录的时刻（毫秒），所以不用另外记账。
    func resendStragglers() {
        guard WCSession.default.activationState == .activated else { return }
        let queued = Set(WCSession.default.outstandingFileTransfers.map { $0.file.fileURL.lastPathComponent })
        let files = (try? FileManager.default.contentsOfDirectory(at: Recorder.spool, includingPropertiesForKeys: [.contentModificationDateKey])) ?? []
        for url in files where url.pathExtension == "m4a" && !queued.contains(url.lastPathComponent) {
            // 刚写了几秒的是正在录的那一段，不碰
            let touched = (try? url.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
            guard Date().timeIntervalSince(touched) > 90, let millis = Double(url.deletingPathExtension().lastPathComponent) else { continue }
            WCSession.default.transferFile(url, metadata: [Wire.startedAt: millis / 1000, Wire.device: "watch"])
        }
        refresh()
    }

    private func refresh() {
        let files = (try? FileManager.default.contentsOfDirectory(at: Recorder.spool, includingPropertiesForKeys: nil)) ?? []
        let count = files.filter { $0.pathExtension == "m4a" }.count
        DispatchQueue.main.async { self.waiting = count }
    }

    // MARK: WCSessionDelegate

    func session(_ session: WCSession, activationDidCompleteWith activationState: WCSessionActivationState, error: Error?) {
        if activationState == .activated { resendStragglers() }
    }

    func session(_ session: WCSession, didFinish fileTransfer: WCSessionFileTransfer, error: Error?) {
        if error == nil { try? FileManager.default.removeItem(at: fileTransfer.file.fileURL) }     // 手机收到了，这一份才删
        refresh()
    }
}
