import Foundation
import Network
import Observation

/// 在局域网里找开着收录音端口的 Mac（Bonjour）。找到的只是一个服务名；要连它，还得解析出地址。
@Observable
final class Finder {
    struct Mac: Identifiable, Hashable {
        let name: String
        let endpoint: NWEndpoint
        var id: String { name }
    }

    private(set) var macs: [Mac] = []
    private var browser: NWBrowser?

    func start() {
        guard browser == nil else { return }
        let browser = NWBrowser(for: .bonjour(type: Wire.bonjourType, domain: nil), using: .tcp)
        browser.browseResultsChangedHandler = { [weak self] results, _ in
            let found = results.compactMap { result -> Mac? in
                if case let .service(name, _, _, _) = result.endpoint { return Mac(name: name, endpoint: result.endpoint) }
                return nil
            }
            DispatchQueue.main.async { self?.macs = found.sorted { $0.name < $1.name } }
        }
        browser.start(queue: .main)
        self.browser = browser
    }

    func stop() {
        browser?.cancel()
        browser = nil
    }

    /// 把服务名解析成"地址:端口"：连一下，看系统实际连到了哪。只要 IPv4——局域网的 IPv6 地址带网卡名，放进网址里很啰嗦。
    static func resolve(_ endpoint: NWEndpoint) async -> (host: String, port: Int)? {
        let parameters = NWParameters.tcp
        if let ip = parameters.defaultProtocolStack.internetProtocol as? NWProtocolIP.Options { ip.version = .v4 }
        let connection = NWConnection(to: endpoint, using: parameters)
        let once = Once()
        return await withCheckedContinuation { (continuation: CheckedContinuation<(host: String, port: Int)?, Never>) in
            let finish: @Sendable ((String, Int)?) -> Void = { value in
                guard once.claim() else { return }           // 连上、失败、超时都会来这里：只认第一个
                connection.cancel()
                continuation.resume(returning: value.map { (host: $0.0, port: $0.1) })
            }
            connection.stateUpdateHandler = { state in
                switch state {
                case .ready:
                    if case let .hostPort(host, port) = connection.currentPath?.remoteEndpoint {
                        var text = "\(host)"
                        if let cut = text.firstIndex(of: "%") { text = String(text[..<cut]) }
                        finish((text, Int(port.rawValue)))
                    } else {
                        finish(nil)
                    }
                case .failed, .cancelled:
                    finish(nil)
                default:
                    break
                }
            }
            connection.start(queue: .main)
            DispatchQueue.main.asyncAfter(deadline: .now() + 6) { finish(nil) }
        }
    }
}

/// 只放行一次的小闸门。
private final class Once: @unchecked Sendable {
    private let lock = NSLock()
    private var done = false

    func claim() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        if done { return false }
        done = true
        return true
    }
}
