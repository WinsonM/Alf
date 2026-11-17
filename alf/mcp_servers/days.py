"""MCP 服务器：今天是什么日子。农历、节气、传统节日，以及中国和澳大利亚新南威尔士州的公共假期。

全部由日历库算出来，不靠模型记忆——模型说农历日期经常出错。
"""
from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import holidays
from lunar_python import Solar
from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

WEEKDAY = "一二三四五六日"
REGION_NAMES = {"CN": "中国", "AU-NSW": "澳大利亚（新州）"}
server = MCPServer("alf-days")


def notable(d: date, regions: list[str]) -> list[str]:
    """这一天有什么特别的；平常日子返回空列表。"""
    solar = Solar.fromYmd(d.year, d.month, d.day)
    lunar = solar.getLunar()
    out = list(lunar.getFestivals()) + list(solar.getFestivals())
    if term := lunar.getJieQi():
        out.append(f"节气：{term}")
    for region in regions:
        country, _, subdiv = region.partition("-")
        try:
            table = holidays.country_holidays(country, subdiv=subdiv or None, years=d.year,
                                              language="zh_CN" if country == "CN" else None)
        except (NotImplementedError, KeyError):
            continue
        if name := table.get(d):
            where = REGION_NAMES.get(region, region)
            same = next((i for i, x in enumerate(out) if x in name or name in x), None)
            if same is None:
                out.append(f"{where}公共假期：{name}")
            else:   # 节日本身已经列出来了（除夕 / 农历除夕），只补一句"放假"
                out[same] = f"{out[same]}（{where}公共假期）"
    return list(dict.fromkeys(out))


def describe(today: date, days_ahead: int = 7, regions: list[str] | None = None) -> dict:
    regions = regions or ["CN", "AU-NSW"]
    lunar = Solar.fromYmd(today.year, today.month, today.day).getLunar()
    upcoming = []
    for k in range(1, days_ahead + 1):
        d = today + timedelta(days=k)
        if what := notable(d, regions):
            upcoming.append({"date": d.isoformat(), "weekday": f"周{WEEKDAY[d.weekday()]}", "in_days": k, "what": what})
    return {"date": today.isoformat(), "weekday": f"周{WEEKDAY[today.weekday()]}",
            "lunar": f"农历{lunar.getMonthInChinese()}月{lunar.getDayInChinese()}",
            "today": notable(today, regions), "upcoming": upcoming}


@server.tool(annotations=ToolAnnotations(read_only_hint=True))
def special_days(days_ahead: int = 7) -> dict:
    """今天的农历日期、节气、节日和公共假期，以及未来几天里的这类日子。"""
    today = datetime.now(ZoneInfo(os.getenv("ALF_TZ", "Australia/Sydney"))).date()
    regions = [r.strip() for r in os.getenv("ALF_HOLIDAY_REGIONS", "CN,AU-NSW").split(",") if r.strip()]
    return describe(today, max(0, min(days_ahead, 30)), regions)


if __name__ == "__main__":
    server.run()
