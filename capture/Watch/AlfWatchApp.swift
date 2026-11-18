import SwiftUI

@main
struct AlfWatchApp: App {
    @State private var recorder = Recorder()
    @State private var courier = Courier()

    var body: some Scene {
        WindowGroup {
            WatchView()
                .environment(recorder)
                .environment(courier)
                .onAppear {
                    recorder.onChunk = { [courier] url, start in courier.send(url, startedAt: start) }
                    courier.activate()
                }
                .onChange(of: recorder.phase) { _, _ in courier.resendStragglers() }      // 每次开始、停下，顺手把落下的补送一遍
        }
    }
}
