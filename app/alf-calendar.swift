import EventKit
import Foundation

// 读写本机「日历」和「提醒事项」，把结果以 JSON 写到指定文件。
// 用法：alf-calendar <输出文件> <命令> [参数…]
//   events <天数> [ask]          从今天起 N 天内的日程
//   reminders <天数> [ask]       N 天内到期（含已逾期）且还没完成的提醒事项
//   add-event <输入.json>        新建日程：{"title", "start", "end", "all_day", "location"}
//   delete-event <标识>          删除日程——只删 Alf 自己建的
//   add-reminder <输入.json>     新建提醒事项：{"title", "due", "has_time"}
//   delete-reminder <标识>       删除提醒事项——只删 Alf 自己建的
//
// 不带 ask 时只检查有没有权限、绝不弹窗——后台每隔几分钟来读一次，不能每次都弹一个授权框。
// 带 ask 时才向用户申请（用户在设置里点"授权"的那一刻）。
//
// 它被包成一个没有界面的小 App（AlfCalendar.app），由系统启动。macOS 只让"正式的 App"申请这两项权限，
// 而且授权是记在发起请求的那个 App 名下的：直接从后台服务里运行它，系统会不弹窗就拒绝。

let arguments = Array(CommandLine.arguments.dropFirst())
let outputPath: String? = arguments.first
let command = arguments.count > 1 ? arguments[1] : "events"
let rest = Array(arguments.dropFirst(2))
// Alf 建的条目都在备注里留这个记号。删除时核对它：就算上层传错了标识，也删不到用户自己的日程。
let mark = "由 Alf 添加"

func finish(_ object: [String: Any], code: Int32) -> Never {
    let data = (try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])) ?? Data("{}".utf8)
    let text = String(data: data, encoding: .utf8) ?? "{}"
    if let outputPath { try? text.write(toFile: outputPath, atomically: true, encoding: .utf8) } else { print(text) }
    exit(code)
}
func fail(_ reason: String, code: Int32 = 1) -> Never { finish(["error": reason], code: code) }

final class Box<T> { var value: T; init(_ value: T) { self.value = value } }

let store = EKEventStore()

func ensureAccess(_ entity: EKEntityType, mayAsk: Bool) {
    let status = EKEventStore.authorizationStatus(for: entity)
    let granted = Box(false)
    if #available(macOS 14.0, *) { granted.value = status == .fullAccess } else { granted.value = status == .authorized }
    if !granted.value && status == .notDetermined {
        guard mayAsk else { fail("not_determined", code: 4) }
        let done = DispatchSemaphore(value: 0)
        if #available(macOS 14.0, *) {
            if entity == .event {
                store.requestFullAccessToEvents { ok, _ in granted.value = ok; done.signal() }
            } else {
                store.requestFullAccessToReminders { ok, _ in granted.value = ok; done.signal() }
            }
        } else {
            store.requestAccess(to: entity) { ok, _ in granted.value = ok; done.signal() }
        }
        if done.wait(timeout: .now() + 75) == .timedOut { fail("timeout", code: 2) }
    }
    guard granted.value else { fail("denied", code: 3) }
}

let iso = ISO8601DateFormatter()
iso.formatOptions = [.withInternetDateTime]
iso.timeZone = .current

func readInput() -> [String: Any] {
    guard let path = rest.first, let data = FileManager.default.contents(atPath: path),
          let input = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { fail("bad_input") }
    return input
}
func date(_ value: Any?) -> Date? { (value as? String).flatMap { iso.date(from: $0) } }
func endOfRange(_ days: Int) -> Date {
    Calendar.current.date(byAdding: .day, value: days, to: Calendar.current.startOfDay(for: Date()))!
}

switch command {
case "events":
    ensureAccess(.event, mayAsk: rest.contains("ask"))
    let start = Calendar.current.startOfDay(for: Date())
    // predicateForEvents 会把重复日程展开成每一次，这正是用 EventKit 而不是 AppleScript 的原因。
    let events = store.events(matching: store.predicateForEvents(withStart: start, end: endOfRange(Int(rest.first ?? "2") ?? 2), calendars: nil))
    let rows: [[String: Any]] = events.sorted { $0.startDate < $1.startDate }.map { e in
        ["id": e.eventIdentifier ?? "", "title": e.title ?? "", "start": iso.string(from: e.startDate), "end": iso.string(from: e.endDate),
         "all_day": e.isAllDay, "calendar": e.calendar.title, "location": e.location ?? "", "recurring": e.hasRecurrenceRules,
         "by_alf": (e.notes ?? "").contains(mark)]
    }
    finish(["events": rows], code: 0)

case "reminders":
    ensureAccess(.reminder, mayAsk: rest.contains("ask"))
    let found = Box<[EKReminder]>([])
    let done = DispatchSemaphore(value: 0)
    let predicate = store.predicateForIncompleteReminders(withDueDateStarting: nil, ending: endOfRange(Int(rest.first ?? "7") ?? 7), calendars: nil)
    store.fetchReminders(matching: predicate) { items in found.value = items ?? []; done.signal() }
    if done.wait(timeout: .now() + 30) == .timedOut { fail("timeout", code: 2) }
    var rows: [[String: Any]] = []
    for r in found.value {
        guard let parts = r.dueDateComponents, let due = Calendar.current.date(from: parts) else { continue }
        rows.append(["id": r.calendarItemIdentifier, "title": r.title ?? "", "due": iso.string(from: due), "has_time": parts.hour != nil,
                     "list": r.calendar.title, "by_alf": (r.notes ?? "").contains(mark)])
    }
    rows.sort { ($0["due"] as! String) < ($1["due"] as! String) }
    finish(["reminders": Array(rows.prefix(50))], code: 0)

case "add-event":
    let input = readInput()
    guard let title = input["title"] as? String, !title.isEmpty, let start = date(input["start"]), let end = date(input["end"]), end >= start else { fail("bad_input") }
    ensureAccess(.event, mayAsk: false)
    guard let calendar = store.defaultCalendarForNewEvents else { fail("no_calendar") }
    let event = EKEvent(eventStore: store)
    event.title = title
    event.startDate = start
    event.endDate = end
    event.isAllDay = (input["all_day"] as? Bool) ?? false
    if let location = input["location"] as? String, !location.isEmpty { event.location = location }
    event.notes = mark
    event.calendar = calendar
    do { try store.save(event, span: .thisEvent, commit: true) } catch { fail("save_failed: \(error.localizedDescription)") }
    finish(["id": event.eventIdentifier ?? "", "calendar": calendar.title], code: 0)

case "delete-event":
    ensureAccess(.event, mayAsk: false)
    guard let id = rest.first, let event = store.event(withIdentifier: id) else { fail("not_found") }
    guard (event.notes ?? "").contains(mark) else { fail("not_by_alf") }
    do { try store.remove(event, span: .thisEvent, commit: true) } catch { fail("remove_failed: \(error.localizedDescription)") }
    finish(["deleted": id], code: 0)

case "add-reminder":
    let input = readInput()
    guard let title = input["title"] as? String, !title.isEmpty else { fail("bad_input") }
    ensureAccess(.reminder, mayAsk: false)
    guard let list = store.defaultCalendarForNewReminders() else { fail("no_calendar") }
    let reminder = EKReminder(eventStore: store)
    reminder.title = title
    reminder.notes = mark
    reminder.calendar = list
    if let due = date(input["due"]) {
        let timed = (input["has_time"] as? Bool) ?? true
        reminder.dueDateComponents = Calendar.current.dateComponents(timed ? [.year, .month, .day, .hour, .minute] : [.year, .month, .day], from: due)
        if timed { reminder.addAlarm(EKAlarm(absoluteDate: due)) }   // 带钟点的，到点由「提醒事项」自己响一次
    }
    do { try store.save(reminder, commit: true) } catch { fail("save_failed: \(error.localizedDescription)") }
    finish(["id": reminder.calendarItemIdentifier, "list": list.title], code: 0)

case "delete-reminder":
    ensureAccess(.reminder, mayAsk: false)
    guard let id = rest.first, let reminder = store.calendarItem(withIdentifier: id) as? EKReminder else { fail("not_found") }
    guard (reminder.notes ?? "").contains(mark) else { fail("not_by_alf") }
    do { try store.remove(reminder, commit: true) } catch { fail("remove_failed: \(error.localizedDescription)") }
    finish(["deleted": id], code: 0)

default:
    fail("unknown_command")
}
