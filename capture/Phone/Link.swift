import CryptoKit
import Foundation
import Observation
import Security

/// 和 Mac 的配对：Mac 在哪（主机名、端口）、它的证书指纹、交录音用的口令。
/// 口令放钥匙串，其余放 UserDefaults。
@Observable
final class Link {
    private(set) var host: String?
    private(set) var port = Wire.defaultPort
    private(set) var pin: String?                // Mac 那张证书的 SHA-256；以后只认这一张
    private(set) var paired = false

    private static let account = "alf-intake-token"
    private let defaults = UserDefaults.standard

    init() {
        host = defaults.string(forKey: "host")
        pin = defaults.string(forKey: "pin")
        if defaults.integer(forKey: "port") > 0 { port = defaults.integer(forKey: "port") }
        paired = host != nil && pin != nil && token != nil
    }

    var token: String? {
        let query: [String: Any] = [kSecClass as String: kSecClassGenericPassword, kSecAttrAccount as String: Self.account,
                                    kSecReturnData as String: true, kSecMatchLimit as String: kSecMatchLimitOne]
        var out: AnyObject?
        guard SecItemCopyMatching(query as CFDictionary, &out) == errSecSuccess, let data = out as? Data else { return nil }
        return String(data: data, encoding: .utf8)
    }

    var base: URL? {
        guard let host else { return nil }
        var parts = URLComponents()
        parts.scheme = "https"
        parts.host = host
        parts.port = port
        return parts.url
    }

    /// 记下这次配对。口令存不进钥匙串就不算配上（返回 false）：存在别处不如不存。
    func save(host: String, port: Int, pin: String, token: String) -> Bool {
        forgetToken()
        // 手机锁着的时候后台也要能传，所以是"解锁过一次之后可读"；不随备份迁到别的设备
        let item: [String: Any] = [kSecClass as String: kSecClassGenericPassword, kSecAttrAccount as String: Self.account,
                                   kSecValueData as String: Data(token.utf8),
                                   kSecAttrAccessible as String: kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly]
        guard SecItemAdd(item as CFDictionary, nil) == errSecSuccess else { return false }
        defaults.set(host, forKey: "host")
        defaults.set(port, forKey: "port")
        defaults.set(pin, forKey: "pin")
        self.host = host
        self.port = port
        self.pin = pin
        paired = true
        return true
    }

    func forget() {
        forgetToken()
        for key in ["host", "port", "pin"] { defaults.removeObject(forKey: key) }
        host = nil
        pin = nil
        port = Wire.defaultPort
        paired = false
    }

    private func forgetToken() {
        SecItemDelete([kSecClass as String: kSecClassGenericPassword, kSecAttrAccount as String: Self.account] as CFDictionary)
    }

    /// 给人核对用的短码，和 Mac 设置页上显示的是同一种算法：指纹的头八位，分成两组。
    static func checkCode(_ pin: String) -> String {
        let head = pin.prefix(8).uppercased()
        return "\(head.prefix(4))-\(head.suffix(4))"
    }
}

/// 只认一张证书的连接。Mac 的证书是它自己签的，系统默认不信；这里不看是谁签的，只看是不是配对时记下的那一张。
/// expected 为空是配对的那一下：第一次见面，先记下看到的指纹（seen），由人核对短码。
final class PinnedSession: NSObject, URLSessionDelegate {
    private let expected: String?
    private(set) var seen: String?
    private(set) lazy var session: URLSession = {
        let config = URLSessionConfiguration.ephemeral
        config.timeoutIntervalForRequest = 20
        config.timeoutIntervalForResource = 120
        config.waitsForConnectivity = false
        return URLSession(configuration: config, delegate: self, delegateQueue: nil)
    }()

    init(expected: String?) { self.expected = expected }

    func urlSession(_ session: URLSession, didReceive challenge: URLAuthenticationChallenge) async
        -> (URLSession.AuthChallengeDisposition, URLCredential?) {
        guard challenge.protectionSpace.authenticationMethod == NSURLAuthenticationMethodServerTrust,
              let trust = challenge.protectionSpace.serverTrust,
              let chain = SecTrustCopyCertificateChain(trust) as? [SecCertificate], let leaf = chain.first else {
            return (.cancelAuthenticationChallenge, nil)
        }
        let print = SHA256.hash(data: SecCertificateCopyData(leaf) as Data).map { String(format: "%02x", $0) }.joined()
        seen = print
        if let expected, expected != print { return (.cancelAuthenticationChallenge, nil) }      // 不是那张证书：有人冒充，或者 Mac 重装过
        return (.useCredential, URLCredential(trust: trust))
    }
}
