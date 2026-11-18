import Foundation

/// 手表、手机、Mac 三端说好的几个名字。改了这里，三端要一起改（Mac 端在 alf/intake.py）。
enum Wire {
    /// 手表传文件给手机时带的两样东西：这一段是什么时候开始录的（墙上时间，秒）、哪个设备录的。
    static let startedAt = "startedAt"
    static let device = "device"
    /// Mac 在局域网里登记的服务名，手机靠它找到 Mac。
    static let bonjourType = "_alf-intake._tcp"
    static let defaultPort = 8766
}

extension URL {
    /// App 自己的一个子目录，没有就建。录音在传走之前放在这里。
    static func appFolder(_ name: String) -> URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        let dir = base.appendingPathComponent(name, isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }
}
