import Foundation
import WatchConnectivity

/// 接手表送来的录音，放进"待传"的文件夹。手机不转写、不听，只是中转：录音在这里等着回到家的 Wi-Fi。
final class Relay: NSObject, WCSessionDelegate {
    static let outbox = URL.appFolder("outbox")

    /// 又到了一段：该试着往 Mac 传了。
    var onFile: (() -> Void)?

    func activate() {
        guard WCSession.isSupported() else { return }
        WCSession.default.delegate = self
        WCSession.default.activate()
    }

    func session(_ session: WCSession, didReceive file: WCSessionFile) {
        // 这个方法一返回，系统就把临时文件删了：必须当场挪走。文件名里记下开始录的时刻和设备，传给 Mac 时要用。
        let started = file.metadata?[Wire.startedAt] as? Double ?? Date().timeIntervalSince1970
        let device = (file.metadata?[Wire.device] as? String ?? "watch").filter { $0.isLetter || $0.isNumber }
        let dest = Self.outbox.appendingPathComponent("\(Int64(started * 1000))_\(device.isEmpty ? "watch" : device).m4a")
        try? FileManager.default.removeItem(at: dest)
        do {
            try FileManager.default.moveItem(at: file.fileURL, to: dest)
        } catch {
            return
        }
        DispatchQueue.main.async { self.onFile?() }
    }

    func session(_ session: WCSession, activationDidCompleteWith activationState: WCSessionActivationState, error: Error?) {}
    func sessionDidBecomeInactive(_ session: WCSession) {}
    func sessionDidDeactivate(_ session: WCSession) { session.activate() }      // 换了一块表：重新接上
}
