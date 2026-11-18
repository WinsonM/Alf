import BackgroundTasks
import SwiftUI

/// 整个 App 就三样东西：和 Mac 的配对（Link）、接手表文件的 Relay、往 Mac 传的 Uploader。
/// 放在一个地方建好：手表送文件来时系统会在后台把 App 叫起来，那时候没有界面，这些也得在。
@MainActor
final class Hub {
    static let shared = Hub()
    static let refreshTask = "alf.capture.upload"

    let link = Link()
    let relay = Relay()
    let uploader: Uploader

    private init() {
        uploader = Uploader(link: link)
        relay.onFile = { [uploader] in uploader.kick() }      // 手表每送来一段，就试着往 Mac 传一轮
        relay.activate()
    }

    /// 请系统过一阵在后台叫我们一次：人回到家却没再录音的时候，靠它把攒着的传走。系统给不给、什么时候给，它说了算。
    func scheduleRefresh() {
        let request = BGAppRefreshTaskRequest(identifier: Self.refreshTask)
        request.earliestBeginDate = Date(timeIntervalSinceNow: 15 * 60)
        try? BGTaskScheduler.shared.submit(request)
    }
}

@main
struct AlfCaptureApp: App {
    @Environment(\.scenePhase) private var phase
    private let hub = Hub.shared

    var body: some Scene {
        WindowGroup {
            PhoneView()
                .environment(hub.link)
                .environment(hub.uploader)
        }
        .onChange(of: phase) { _, now in
            if now == .active { hub.uploader.kick() }
            if now == .background { hub.scheduleRefresh() }
        }
        .backgroundTask(.appRefresh(Hub.refreshTask)) {
            await Hub.shared.scheduleRefresh()
            await Hub.shared.uploader.drainNow()
        }
    }
}
