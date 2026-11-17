import Cocoa

// 画 Alf 的图标：和网页顶栏那个圆角方块一样的蓝色渐变，中间一个白色的 A。用法：swift make_icon.swift 输出.png

let size: CGFloat = 1024
let image = NSImage(size: NSSize(width: size, height: size))
image.lockFocus()

let inset: CGFloat = 100                     // macOS 图标四周要留白
let tile = NSRect(x: inset, y: inset, width: size - inset * 2, height: size - inset * 2)
let shape = NSBezierPath(roundedRect: tile, xRadius: 185, yRadius: 185)
NSGradient(colors: [
    NSColor(srgbRed: 0.35, green: 0.78, blue: 0.98, alpha: 1),   // #5ac8fa
    NSColor(srgbRed: 0.04, green: 0.52, blue: 1.00, alpha: 1),   // #0a84ff
    NSColor(srgbRed: 0.37, green: 0.36, blue: 0.90, alpha: 1),   // #5e5ce6
])!.draw(in: shape, angle: -50)

let letter = NSAttributedString(string: "A", attributes: [
    .font: NSFont.systemFont(ofSize: 520, weight: .bold),
    .foregroundColor: NSColor.white,
])
let bounds = letter.size()
letter.draw(at: NSPoint(x: (size - bounds.width) / 2, y: (size - bounds.height) / 2 + 10))

image.unlockFocus()
let png = NSBitmapImageRep(data: image.tiffRepresentation!)!.representation(using: .png, properties: [:])!
try! png.write(to: URL(fileURLWithPath: CommandLine.arguments[1]))
