import Foundation
import Observation
import UIKit

/// 把"待传"文件夹里的录音一段一段交给 Mac。交成了才删；Mac 不在（不在家、Mac 睡了）就停下，下次再试。
@Observable
@MainActor
final class Uploader {
    private(set) var pending = 0
    private(set) var lastUpload: Date?
    private(set) var problem: String?
    private(set) var busy = false

    private let link: Link

    init(link: Link) {
        self.link = link
        refresh()
    }

    func refresh() {
        pending = files().count
    }

    private func files() -> [URL] {
        let all = (try? FileManager.default.contentsOfDirectory(at: Relay.outbox, includingPropertiesForKeys: nil)) ?? []
        return all.filter { $0.pathExtension == "m4a" }.sorted { $0.lastPathComponent < $1.lastPathComponent }      // 文件名以时刻开头：早的先传
    }

    /// 试着传一轮。已经在传就不重复起。
    func kick() {
        refresh()
        guard !busy, link.paired, pending > 0 else { return }
        busy = true
        Task {
            // 手表送文件来的时候 App 多半在后台：向系统要一小段时间把手头的传完
            let background = UIApplication.shared.beginBackgroundTask(withName: "alf-upload")
            await drain()
            busy = false
            refresh()
            UIApplication.shared.endBackgroundTask(background)
        }
    }

    /// 给系统的后台刷新用：传一轮，传完才返回。
    func drainNow() async {
        guard !busy, link.paired else { return }
        busy = true
        await drain()
        busy = false
        refresh()
    }

    private func drain() async {
        guard let base = link.base, let token = link.token, let pin = link.pin else { return }
        let pinned = PinnedSession(expected: pin)
        defer { pinned.session.finishTasksAndInvalidate() }
        for url in files() {
            let name = url.deletingPathExtension().lastPathComponent.split(separator: "_", maxSplits: 1)
            guard let millis = Double(name.first ?? "") else {
                try? FileManager.default.removeItem(at: url)
                continue
            }
            var request = URLRequest(url: base.appendingPathComponent("recordings"))
            request.httpMethod = "PUT"
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
            request.setValue(String(millis / 1000), forHTTPHeaderField: "X-Started-At")
            request.setValue(name.count > 1 ? String(name[1]) : "watch", forHTTPHeaderField: "X-Device")
            request.setValue("audio/mp4", forHTTPHeaderField: "Content-Type")
            do {
                let (_, response) = try await pinned.session.upload(for: request, fromFile: url)
                let code = (response as? HTTPURLResponse)?.statusCode ?? 0
                switch code {
                case 200, 201:
                    try? FileManager.default.removeItem(at: url)
                    lastUpload = Date()
                    problem = nil
                    pending = max(0, pending - 1)
                case 401:
                    problem = "Mac 不认这台手机了，要重新配对"
                    return
                case 413, 422:
                    try? FileManager.default.removeItem(at: url)      // Mac 说这一段它不收（坏文件、太旧）：留着也没用
                    problem = "有一段录音 Mac 没收"
                default:
                    problem = "Mac 那边出了点问题（\(code)）"
                    return
                }
            } catch {
                // 多半只是不在家，或者 Mac 睡了：不算毛病，照实说
                problem = (error as? URLError)?.code == .cancelled ? "Mac 的证书对不上，可能要重新配对" : "现在连不上 Mac，回到家里的 Wi-Fi 再传"
                return
            }
        }
    }

    // MARK: 配对

    /// 拿配对码去换口令。成功返回核对码（给人和 Mac 上显示的对一眼）；失败抛出给人看的原因。
    func pair(host: String, port: Int, code: String) async throws -> String {
        var parts = URLComponents()
        parts.scheme = "https"
        parts.host = host
        parts.port = port
        parts.path = "/pair"
        guard let url = parts.url else { throw PairError("地址不对") }
        let pinned = PinnedSession(expected: nil)
        defer { pinned.session.finishTasksAndInvalidate() }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: ["code": code])
        let data: Data, response: URLResponse
        do {
            (data, response) = try await pinned.session.data(for: request)
        } catch {
            throw PairError("连不上这台 Mac。手机和它在同一个 Wi-Fi 上吗？")
        }
        let status = (response as? HTTPURLResponse)?.statusCode ?? 0
        let body = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] ?? [:]
        guard status == 200, let token = body["token"] as? String, let seen = pinned.seen else {
            throw PairError(body["detail"] as? String ?? "没配上（\(status)）")
        }
        // Mac 告诉我们它在局域网里的名字（xxx.local）：以后用名字找它，路由器给它换了 IP 也不怕
        guard link.save(host: body["host"] as? String ?? host, port: body["port"] as? Int ?? port, pin: seen, token: token) else {
            throw PairError("口令没能存进这部手机的钥匙串，没配上")
        }
        kick()
        return Link.checkCode(seen)
    }

    func unpair() {
        link.forget()
        problem = nil
    }
}

struct PairError: LocalizedError {
    let why: String
    init(_ why: String) { self.why = why }
    var errorDescription: String? { why }
}
