#!/bin/bash
# 编译 Alf.app（原生窗口外壳）。需要装了 Xcode 或命令行工具。
# Alf.app 直接放在项目文件夹的第一层：打开文件夹就能双击启动，不用去 dist/ 里找。
# 没有界面的日历小程序仍然在 dist/ 里，那个不需要人去点。
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=Alf.app
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

rm -rf "$OUT"
mkdir -p "$OUT/Contents/MacOS" "$OUT/Contents/Resources"

swiftc -O -o "$OUT/Contents/MacOS/Alf" app/Alf.swift -framework Cocoa -framework WebKit
cp app/Info.plist "$OUT/Contents/Info.plist"

# 图标：画一张 1024 的大图，缩成各种尺寸，打包成 icns。
swift app/make_icon.swift "$TMP/icon.png"
mkdir "$TMP/AppIcon.iconset"
for s in 16 32 128 256 512; do
  sips -z $s $s "$TMP/icon.png" --out "$TMP/AppIcon.iconset/icon_${s}x${s}.png" >/dev/null
  sips -z $((s * 2)) $((s * 2)) "$TMP/icon.png" --out "$TMP/AppIcon.iconset/icon_${s}x${s}@2x.png" >/dev/null
done
iconutil -c icns "$TMP/AppIcon.iconset" -o "$OUT/Contents/Resources/AppIcon.icns"

# 本机签名：让系统把它当成一个固定的 App（麦克风授权会记在它名下）。
codesign --force --sign - "$OUT" 2>/dev/null

# 读日历的小程序，包成一个没有界面的 App。后台服务通过 MCP 服务器、经由系统的 open 来启动它，
# 这样日历授权会以"Alf 日历助手"的名义弹出来并记住。
CAL=dist/AlfCalendar.app
rm -rf "$CAL" dist/alf-calendar
mkdir -p "$CAL/Contents/MacOS"
swiftc -O -o "$CAL/Contents/MacOS/alf-calendar" app/alf-calendar.swift
cp app/calendar-Info.plist "$CAL/Contents/Info.plist"
codesign --force --sign - "$CAL" 2>/dev/null

echo "已生成 $OUT 和 $CAL"
