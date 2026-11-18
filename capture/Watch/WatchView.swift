import SwiftUI

/// 手表上只有一屏：一颗大按钮，录没录一眼看得出；下面是录了多久、还有几段没送到手机。
struct WatchView: View {
    @Environment(Recorder.self) private var recorder
    @Environment(Courier.self) private var courier

    private var live: Bool { recorder.phase == .recording }

    var body: some View {
        VStack(spacing: 8) {
            Button(action: recorder.toggle) {
                ZStack {
                    Circle()
                        .fill(live ? Color.red.opacity(0.22) : Color.white.opacity(0.08))
                        .scaleEffect(live ? 1 + CGFloat(recorder.level) * 0.22 : 1)       // 有人说话时那一圈跟着鼓起来：一眼知道它在听
                        .animation(.spring(response: 0.25, dampingFraction: 0.7), value: recorder.level)
                    Circle()
                        .fill(live ? Color.red : Color.white.opacity(0.9))
                        .padding(live ? 26 : 18)
                        .animation(.spring(response: 0.35, dampingFraction: 0.75), value: live)
                    Image(systemName: live ? "pause.fill" : "mic.fill")
                        .font(.system(size: 26, weight: .semibold))
                        .foregroundStyle(live ? .white : .black)
                }
                .frame(width: 104, height: 104)
            }
            .buttonStyle(.plain)
            .accessibilityLabel(live ? "暂停录音" : "开始录音")

            status
                .font(.footnote)
                .multilineTextAlignment(.center)

            if courier.waiting > 0 {
                Text("\(courier.waiting) 段等着传给手机")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.horizontal, 6)
    }

    @ViewBuilder
    private var status: some View {
        switch recorder.phase {
        case .idle:
            Text("轻点开始录").foregroundStyle(.secondary)
        case .recording:
            if let since = recorder.since {
                // 系统自己走的计时：不用每秒叫醒 App
                Text(since, style: .timer).monospacedDigit().foregroundStyle(.red)
            }
        case .interrupted:
            Text("被打断了，结束后接着录").foregroundStyle(.orange)
        case .failed(let why):
            Text(why).foregroundStyle(.orange)
        }
    }
}
