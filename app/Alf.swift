import Cocoa
import WebKit

// Alf 的原生窗口外壳：一个窗口，里面显示后台服务提供的页面。
//
// 所有功能都在后台服务里（那个由 launchd 常驻的 Python 进程），这里只负责"像个 App"：
// 程序坞图标、独立窗口、复制粘贴等菜单、麦克风授权，以及后台没起来时给一句人话的提示。
// 关掉这个窗口不影响后台：提醒照发，iMessage 照回。

let defaultAddress = "http://127.0.0.1:8765/"

let offlinePage = """
<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<style>
  html,body{height:100%;margin:0}
  body{display:grid;place-items:center;font:16px -apple-system,"PingFang SC",sans-serif;background:#eef1f8;color:#101828}
  @media(prefers-color-scheme:dark){body{background:#0b0d13;color:#f2f4f7}}
  div{max-width:420px;text-align:center;padding:24px}
  h1{font-size:22px;margin:0 0 8px} p{color:#667085;margin:6px 0}
  code{font:13px ui-monospace,Menlo,monospace;user-select:all}
</style>
<div><h1>Alf 的后台服务没有在运行</h1>
<p>这个窗口只是界面，真正干活的是后台服务。</p>
<p>在终端里进入项目目录，运行：</p>
<p><code>uv run alf service install</code></p>
<p>服务起来后这里会自动恢复。</p></div></html>
"""

final class AppDelegate: NSObject, NSApplicationDelegate, WKNavigationDelegate, WKUIDelegate {
    private var window: NSWindow!
    private var webView: WKWebView!
    private var retry: Timer?
    private let address: URL

    override init() {
        // --url 可以把窗口指到别的地址（比如另一个端口上的演示实例）。
        let args = CommandLine.arguments
        let custom = args.firstIndex(of: "--url").flatMap { $0 + 1 < args.count ? args[$0 + 1] : nil }
        address = URL(string: custom ?? defaultAddress) ?? URL(string: defaultAddress)!
        super.init()
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        buildMenu()

        let configuration = WKWebViewConfiguration()
        configuration.applicationNameForUserAgent = "AlfApp/1"
        webView = WKWebView(frame: .zero, configuration: configuration)
        webView.navigationDelegate = self
        webView.uiDelegate = self

        window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 900, height: 760),
                          styleMask: [.titled, .closable, .miniaturizable, .resizable],
                          backing: .buffered, defer: false)
        window.title = "Alf"
        window.titleVisibility = .hidden
        window.titlebarAppearsTransparent = true      // 标题栏和页面背景连成一片
        window.backgroundColor = NSColor(name: nil) { appearance in
            appearance.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua
                ? NSColor(srgbRed: 0.043, green: 0.051, blue: 0.075, alpha: 1)    // 页面深色背景 #0b0d13
                : NSColor(srgbRed: 0.933, green: 0.945, blue: 0.973, alpha: 1)    // 页面浅色背景 #eef1f8
        }
        window.minSize = NSSize(width: 420, height: 520)
        window.contentView = webView
        window.center()
        window.setFrameAutosaveName("AlfMainWindow")  // 记住上次的位置和大小
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)

        load()
    }

    private func load() {
        webView.load(URLRequest(url: address, cachePolicy: .reloadIgnoringLocalCacheData, timeoutInterval: 5))
    }

    @objc private func reload() { load() }

    // 页面加载不出来：后台服务没起。显示说明，并每两秒探一次，起来了就自动恢复。
    private func showOffline() {
        webView.loadHTMLString(offlinePage, baseURL: nil)
        retry?.invalidate()
        retry = Timer.scheduledTimer(withTimeInterval: 2, repeats: true) { [weak self] timer in
            guard let self else { return }
            var probe = URLRequest(url: self.address.appendingPathComponent("api/status"))
            probe.timeoutInterval = 1.5
            URLSession.shared.dataTask(with: probe) { _, response, _ in
                guard (response as? HTTPURLResponse)?.statusCode == 200 else { return }
                DispatchQueue.main.async {
                    timer.invalidate()
                    self.load()
                }
            }.resume()
        }
    }

    func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!, withError error: Error) {
        print("failed \(address.absoluteString): \(error.localizedDescription)")
        fflush(stdout)
        showOffline()
    }

    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
        print("loaded \(webView.url?.absoluteString ?? "-") title=\(webView.title ?? "")")
        fflush(stdout)
        snapshotIfAsked()
    }

    // 测试用：`--snapshot 路径` 会在页面加载完两秒后，把窗口里实际画出来的内容存成 PNG 然后退出。
    // 系统截屏抓不到网页内容（它由另一个进程渲染），所以要验证界面只能让它自己拍。
    private func snapshotIfAsked() {
        let args = CommandLine.arguments
        guard let i = args.firstIndex(of: "--snapshot"), i + 1 < args.count else { return }
        let path = args[i + 1]
        DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
            self.webView.takeSnapshot(with: nil) { image, error in
                if let tiff = image?.tiffRepresentation, let png = NSBitmapImageRep(data: tiff)?.representation(using: .png, properties: [:]) {
                    try? png.write(to: URL(fileURLWithPath: path))
                    print("snapshot \(path)")
                } else {
                    print("snapshot failed: \(error?.localizedDescription ?? "unknown")")
                }
                fflush(stdout)
                NSApp.terminate(nil)
            }
        }
    }

    // 只在窗口里打开 Alf 自己的页面；别的链接交给默认浏览器。
    func webView(_ webView: WKWebView, decidePolicyFor navigationAction: WKNavigationAction,
                 decisionHandler: @escaping (WKNavigationActionPolicy) -> Void) {
        if let url = navigationAction.request.url, let scheme = url.scheme, scheme.hasPrefix("http"),
           url.host != address.host {
            NSWorkspace.shared.open(url)
            decisionHandler(.cancel)
            return
        }
        decisionHandler(.allow)
    }

    // 页面上的语音按钮要用麦克风。只给 Alf 自己的页面；系统还会再问用户一次。
    func webView(_ webView: WKWebView, requestMediaCapturePermissionFor origin: WKSecurityOrigin,
                 initiatedByFrame frame: WKFrameInfo, type: WKMediaCaptureType,
                 decisionHandler: @escaping (WKPermissionDecision) -> Void) {
        decisionHandler(origin.host == address.host ? .grant : .deny)
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }

    // 没有"编辑"菜单的话，网页输入框里 ⌘C / ⌘V 不起作用——粘贴 key 就靠它。
    private func buildMenu() {
        let main = NSMenu()

        let appItem = NSMenuItem()
        let appMenu = NSMenu()
        appMenu.addItem(withTitle: "隐藏 Alf", action: #selector(NSApplication.hide(_:)), keyEquivalent: "h")
        appMenu.addItem(.separator())
        appMenu.addItem(withTitle: "退出 Alf", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        appItem.submenu = appMenu
        main.addItem(appItem)

        let editItem = NSMenuItem()
        let edit = NSMenu(title: "编辑")
        edit.addItem(withTitle: "撤销", action: Selector(("undo:")), keyEquivalent: "z")
        edit.addItem(withTitle: "重做", action: Selector(("redo:")), keyEquivalent: "Z")
        edit.addItem(.separator())
        edit.addItem(withTitle: "剪切", action: #selector(NSText.cut(_:)), keyEquivalent: "x")
        edit.addItem(withTitle: "拷贝", action: #selector(NSText.copy(_:)), keyEquivalent: "c")
        edit.addItem(withTitle: "粘贴", action: #selector(NSText.paste(_:)), keyEquivalent: "v")
        edit.addItem(withTitle: "全选", action: #selector(NSText.selectAll(_:)), keyEquivalent: "a")
        editItem.submenu = edit
        main.addItem(editItem)

        let viewItem = NSMenuItem()
        let view = NSMenu(title: "显示")
        let reloadItem = NSMenuItem(title: "重新载入", action: #selector(reload), keyEquivalent: "r")
        reloadItem.target = self
        view.addItem(reloadItem)
        viewItem.submenu = view
        main.addItem(viewItem)

        let windowItem = NSMenuItem()
        let windowMenu = NSMenu(title: "窗口")
        windowMenu.addItem(withTitle: "最小化", action: #selector(NSWindow.performMiniaturize(_:)), keyEquivalent: "m")
        windowMenu.addItem(withTitle: "关闭", action: #selector(NSWindow.performClose(_:)), keyEquivalent: "w")
        windowItem.submenu = windowMenu
        main.addItem(windowItem)
        NSApp.windowsMenu = windowMenu

        NSApp.mainMenu = main
    }
}

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.setActivationPolicy(.regular)
app.run()
