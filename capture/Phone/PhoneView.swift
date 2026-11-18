import SwiftUI

/// 手机上只有一屏：和哪台 Mac 配对了、还有几段没传、上次什么时候传的。没配对时，这一屏就是配对的地方。
struct PhoneView: View {
    @Environment(Link.self) private var link
    @Environment(Uploader.self) private var uploader
    @State private var finder = Finder()
    @State private var target: Target?            // 正在配对的那台 Mac
    @State private var manual = false
    @State private var confirmUnpair = false
    @State private var checked: String?           // 刚配上：给他看核对码
    @State private var reaching: String?          // 正在连哪台 Mac
    @State private var note: String?              // 没连上时的一句话

    struct Target: Identifiable {
        let name: String
        let host: String
        let port: Int
        var id: String { "\(host):\(port)" }
    }

    var body: some View {
        ZStack {
            LinearGradient(colors: [Color(.systemGroupedBackground), Color(.secondarySystemGroupedBackground)], startPoint: .top, endPoint: .bottom)
                .ignoresSafeArea()
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    Text("Alf 录音").font(.largeTitle.bold())
                    Text("手表上点开始，录音会先到这部手机，回到家里的 Wi-Fi 再交给 Mac 上的 Alf。手机不听、不存，只是中转。")
                        .font(.subheadline).foregroundStyle(.secondary)
                    if link.paired { pairedCard } else { pairingCard }
                }
                .padding(20)
            }
            if confirmUnpair { unpairDialog.transition(.scale(scale: 0.92).combined(with: .opacity)) }
            if let checked { checkDialog(checked).transition(.scale(scale: 0.92).combined(with: .opacity)) }
        }
        .animation(.spring(response: 0.35, dampingFraction: 0.8), value: confirmUnpair)
        .animation(.spring(response: 0.35, dampingFraction: 0.8), value: checked)
        .sheet(item: $target) { mac in
            CodeSheet(mac: mac) { code in try await uploader.pair(host: mac.host, port: mac.port, code: code) } done: { check in
                target = nil
                checked = check
            }
            .presentationDetents([.medium])
        }
        .sheet(isPresented: $manual) {
            ManualSheet { host, port in
                manual = false
                target = Target(name: host, host: host, port: port)
            }
            .presentationDetents([.medium])
        }
        .onAppear { if !link.paired { finder.start() }; uploader.kick() }
        .onChange(of: link.paired) { _, paired in paired ? finder.stop() : finder.start() }
    }

    // MARK: 已配对

    private var pairedCard: some View {
        VStack(alignment: .leading, spacing: 14) {
            Label(link.host ?? "", systemImage: "desktopcomputer").font(.headline)
            HStack(alignment: .firstTextBaseline) {
                Text("\(uploader.pending)").font(.system(size: 44, weight: .bold, design: .rounded)).contentTransition(.numericText())
                Text("段等着传").foregroundStyle(.secondary)
                Spacer()
                if uploader.busy { ProgressView() }
            }
            if let last = uploader.lastUpload {
                Text("上次传到 Mac：\(last.formatted(date: .omitted, time: .shortened))").font(.footnote).foregroundStyle(.secondary)
            }
            if let problem = uploader.problem {
                Text(problem).font(.footnote).foregroundStyle(.orange)
            }
            HStack(spacing: 10) {
                Pill("现在传") { uploader.kick() }
                    .disabled(uploader.busy || uploader.pending == 0)
                Pill("解除配对") { confirmUnpair = true }
            }
        }
        .padding(18)
        .frame(maxWidth: .infinity, alignment: .leading)
        .glassBackground(RoundedRectangle(cornerRadius: 24, style: .continuous))
    }

    // MARK: 还没配对

    private var pairingCard: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("先和 Mac 配对").font(.headline)
            Text("在 Mac 上打开 Alf 的设置页，打开「接收手表传来的录音」，点「配对手机」拿到六位数。")
                .font(.footnote).foregroundStyle(.secondary)
            if finder.macs.isEmpty {
                HStack(spacing: 10) { ProgressView(); Text("正在这个 Wi-Fi 上找 Mac……").font(.subheadline).foregroundStyle(.secondary) }
            }
            ForEach(finder.macs) { mac in
                Button {
                    reaching = mac.name
                    note = nil
                    Task {
                        if let found = await Finder.resolve(mac.endpoint) {
                            target = Target(name: mac.name, host: found.host, port: found.port)
                        } else {
                            note = "没连上 \(mac.name)。它还开着吗？也可以试试手动填地址。"
                        }
                        reaching = nil
                    }
                } label: {
                    HStack {
                        Image(systemName: "desktopcomputer")
                        Text(mac.name)
                        Spacer()
                        if reaching == mac.name { ProgressView() } else { Image(systemName: "chevron.right").font(.footnote).foregroundStyle(.secondary) }
                    }
                    .padding(.horizontal, 16).padding(.vertical, 12)
                    .contentShape(Capsule())          // 整条都能点，不只是字
                }
                .buttonStyle(.plain)
                .glassBackground(Capsule(), interactive: true)
                .disabled(reaching != nil)
            }
            if let note { Text(note).font(.footnote).foregroundStyle(.orange) }
            Button("手动填 Mac 的地址") { manual = true }.font(.footnote)
            if uploader.pending > 0 {
                Text("已经有 \(uploader.pending) 段录音在这部手机上等着，配对之后会一起传过去。").font(.footnote).foregroundStyle(.secondary)
            }
        }
        .padding(18)
        .frame(maxWidth: .infinity, alignment: .leading)
        .glassBackground(RoundedRectangle(cornerRadius: 24, style: .continuous))
    }

    // MARK: 弹窗：居中的玻璃卡片 + 压暗的遮罩；点遮罩等于取消

    private func dialog<Content: View>(dismiss: @escaping () -> Void, @ViewBuilder content: () -> Content) -> some View {
        ZStack {
            Color.black.opacity(0.35).ignoresSafeArea().onTapGesture(perform: dismiss)
            VStack(spacing: 14, content: content)
                .padding(22)
                .frame(maxWidth: 320)
                .glassBackground(RoundedRectangle(cornerRadius: 28, style: .continuous))
                .padding(24)
        }
    }

    private var unpairDialog: some View {
        dialog(dismiss: { confirmUnpair = false }) {
            Text("解除配对？").font(.headline)
            Text("这部手机不会再往 Mac 传录音。还没传走的会留在手机上，重新配对之后接着传。")
                .font(.subheadline).foregroundStyle(.secondary).multilineTextAlignment(.center)
            HStack(spacing: 10) {
                Pill("取消", wide: true) { confirmUnpair = false }
                Pill("解除", destructive: true, wide: true) { uploader.unpair(); confirmUnpair = false }
            }
        }
    }

    private func checkDialog(_ code: String) -> some View {
        dialog(dismiss: { checked = nil }) {
            Text("配上了").font(.headline)
            Text("核对码").font(.footnote).foregroundStyle(.secondary)
            Text(code).font(.system(size: 30, weight: .semibold, design: .monospaced))
            Text("Mac 的设置页上显示的应该是同一个。不一样的话，点「解除配对」重来——可能连到了别的机器上。")
                .font(.footnote).foregroundStyle(.secondary).multilineTextAlignment(.center)
            Pill("对得上", wide: true) { checked = nil }
        }
    }
}

/// 胶囊按钮：平常的是玻璃的，破坏性的是红色实心的。整颗胶囊都能点——不只是中间那两个字。
struct Pill: View {
    let title: String
    var destructive = false
    var wide = false
    let action: () -> Void
    @Environment(\.isEnabled) private var enabled

    init(_ title: String, destructive: Bool = false, wide: Bool = false, action: @escaping () -> Void) {
        self.title = title
        self.destructive = destructive
        self.wide = wide
        self.action = action
    }

    var body: some View {
        let label = Button(action: action) {
            Text(title)
                .frame(maxWidth: wide ? .infinity : nil)
                .padding(.horizontal, 18).padding(.vertical, 11)
                .contentShape(Capsule())
        }
        .buttonStyle(.plain)
        .opacity(enabled ? 1 : 0.45)
        if destructive {
            label.foregroundStyle(.white).background(Color.red, in: Capsule())
        } else {
            label.glassBackground(Capsule(), interactive: true)
        }
    }
}

/// 输配对码的那一页。
private struct CodeSheet: View {
    let mac: PhoneView.Target
    let pair: (String) async throws -> String
    let done: (String) -> Void
    @State private var code = ""
    @State private var problem: String?
    @State private var working = false
    @FocusState private var focused: Bool

    var body: some View {
        VStack(spacing: 16) {
            Text("和 \(mac.name) 配对").font(.headline)
            Text("输入 Mac 设置页上的六位数").font(.subheadline).foregroundStyle(.secondary)
            TextField("000000", text: $code)
                .keyboardType(.numberPad)
                .textContentType(.oneTimeCode)
                .font(.system(size: 36, weight: .semibold, design: .monospaced))
                .multilineTextAlignment(.center)
                .focused($focused)
                .onChange(of: code) { _, new in
                    code = String(new.filter(\.isNumber).prefix(6))
                    if code.count == 6 { submit() }
                }
            if let problem { Text(problem).font(.footnote).foregroundStyle(.orange).multilineTextAlignment(.center) }
            if working { ProgressView() }
            Spacer()
        }
        .padding(24)
        .onAppear { focused = true }
    }

    private func submit() {
        guard !working else { return }
        working = true
        problem = nil
        Task {
            do {
                done(try await pair(code))
            } catch {
                problem = error.localizedDescription
                code = ""
            }
            working = false
        }
    }
}

/// 找不到 Mac 的时候（路由器把设备互相隔开了）：手动填地址。
private struct ManualSheet: View {
    let done: (String, Int) -> Void
    @State private var host = ""
    @State private var port = String(Wire.defaultPort)

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("手动填 Mac 的地址").font(.headline)
            Text("Mac 的设置页「配对手机」那一行里写着它的名字（xxx.local）；也可以填它的 IP。").font(.footnote).foregroundStyle(.secondary)
            TextField("study-mac.local", text: $host)
                .textInputAutocapitalization(.never).autocorrectionDisabled().keyboardType(.URL)
                .padding(12).glassBackground(RoundedRectangle(cornerRadius: 14, style: .continuous))
            TextField("端口", text: $port)
                .keyboardType(.numberPad)
                .padding(12).glassBackground(RoundedRectangle(cornerRadius: 14, style: .continuous))
            Pill("下一步", wide: true) { done(host.trimmingCharacters(in: .whitespaces), Int(port) ?? Wire.defaultPort) }
                .disabled(host.trimmingCharacters(in: .whitespaces).isEmpty)
            Spacer()
        }
        .padding(24)
    }
}
