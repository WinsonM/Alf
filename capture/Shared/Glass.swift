import SwiftUI

/// 统一的"液态玻璃背景"。26 系统用正宗的 `.glassEffect()`，更早的系统降级成 `.regularMaterial`。
/// 写法和 Jogy 里的同名修饰符一样，两个 App 看起来是一家的。
extension View {
    @ViewBuilder
    func glassBackground<S: Shape>(_ shape: S, interactive: Bool = false) -> some View {
        if #available(iOS 26.0, watchOS 26.0, *) {
            self.glassEffect(interactive ? .regular.interactive() : .regular, in: shape)
        } else {
            self.background(.regularMaterial, in: shape)
        }
    }
}
