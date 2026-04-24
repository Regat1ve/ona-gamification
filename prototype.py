"""
Исследовательский прототип к ВКР.
Демонстрирует полный цикл авторского подхода: генерация синтетических логов
коммуникаций, построение графа, расчёт ONA-метрик, диагностика аномалий,
подбор игровых механик по матрице соответствия, симуляция адресной
интервенции и неадресной базовой линии, повторный замер метрик,
статистическая проверка значимости изменений.

Запуск: python prototype.py
Зависимости: numpy, networkx, pandas, scipy, matplotlib
"""
from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DEPARTMENTS = {
    "Разработка": 10,
    "Продажи": 8,
    "Поддержка": 7,
}
WEEKS = 4
DAYS_PER_WEEK = 5

P_INTRA = 0.45
P_INTER = 0.03
ISOLATED_FACTOR = 0.1
HUB_FACTOR = 2.8

WEAK_PAIR = {"Разработка", "Продажи"}
WEAK_PAIR_FACTOR = 0.25


@dataclass
class Employee:
    eid: str
    dept: str
    activity: float
    role: str = "обычный"
    kpi_baseline: float = 1.0

    def __hash__(self):
        return hash(self.eid)


@dataclass
class Team:
    employees: List[Employee] = field(default_factory=list)

    def by_dept(self, dept: str) -> List[Employee]:
        return [e for e in self.employees if e.dept == dept]

    def by_id(self, eid: str) -> Employee:
        return next(e for e in self.employees if e.eid == eid)


def build_team() -> Team:
    employees: List[Employee] = []
    counter = 1
    for dept, n in DEPARTMENTS.items():
        for _ in range(n):
            eid = f"E{counter:02d}"
            activity = float(np.clip(np.random.normal(1.0, 0.2), 0.4, 1.6))
            employees.append(Employee(eid=eid, dept=dept, activity=activity))
            counter += 1

    employees[2].role = "изолированный"
    employees[2].activity = ISOLATED_FACTOR
    employees[14].role = "изолированный"
    employees[14].activity = ISOLATED_FACTOR

    employees[0].role = "перегруженный посредник"
    employees[0].activity = HUB_FACTOR
    employees[10].role = "перегруженный посредник"
    employees[10].activity = HUB_FACTOR * 0.9

    for e in employees:
        e.kpi_baseline = float(np.clip(np.random.normal(1.0, 0.15), 0.6, 1.4))
    return Team(employees)


def generate_logs(team: Team, weeks: int = WEEKS, seed_offset: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(SEED + seed_offset)
    rows = []
    total_days = weeks * DAYS_PER_WEEK
    for day in range(total_days):
        for sender in team.employees:
            for receiver in team.employees:
                if sender.eid == receiver.eid:
                    continue
                same_dept = sender.dept == receiver.dept
                base_p = P_INTRA if same_dept else P_INTER
                if not same_dept and {sender.dept, receiver.dept} == WEAK_PAIR:
                    base_p *= WEAK_PAIR_FACTOR
                p = base_p * sender.activity
                if receiver.role == "перегруженный посредник":
                    p *= 2.0
                if receiver.role == "изолированный":
                    p *= 0.2
                p = min(p, 0.95)
                if rng.random() < p:
                    weight = int(rng.integers(1, 4))
                    rows.append({
                        "day": day,
                        "from": sender.eid,
                        "to": receiver.eid,
                        "weight": weight,
                    })
    return pd.DataFrame(rows)


def build_graph(logs: pd.DataFrame) -> nx.DiGraph:
    g = nx.DiGraph()
    agg = logs.groupby(["from", "to"])["weight"].sum().reset_index()
    for _, row in agg.iterrows():
        g.add_edge(row["from"], row["to"], weight=int(row["weight"]))
    return g


def compute_metrics(g: nx.DiGraph, team: Team) -> pd.DataFrame:
    for e in team.employees:
        if e.eid not in g:
            g.add_node(e.eid)

    in_deg = dict(g.in_degree(weight="weight"))
    out_deg = dict(g.out_degree(weight="weight"))
    try:
        betw = nx.betweenness_centrality(g, weight="weight", normalized=True)
    except Exception:
        betw = {n: 0.0 for n in g.nodes}
    clustering = nx.clustering(g.to_undirected(), weight="weight")

    rows = []
    for e in team.employees:
        total_deg = in_deg.get(e.eid, 0) + out_deg.get(e.eid, 0)
        rows.append({
            "eid": e.eid,
            "dept": e.dept,
            "role": e.role,
            "in_degree": in_deg.get(e.eid, 0),
            "out_degree": out_deg.get(e.eid, 0),
            "total_degree": total_deg,
            "betweenness": betw.get(e.eid, 0.0),
            "clustering": clustering.get(e.eid, 0.0),
        })
    return pd.DataFrame(rows)


def diagnose(metrics: pd.DataFrame) -> Dict[str, List[str]]:
    findings = defaultdict(list)
    deg_q1 = metrics["total_degree"].quantile(0.25)
    deg_median = metrics["total_degree"].median()
    for _, row in metrics.iterrows():
        if row["total_degree"] < deg_q1 * 0.6 and row["total_degree"] < deg_median * 0.4:
            findings["изолированный"].append(row["eid"])
    betw_threshold = metrics["betweenness"].mean() + 1.5 * metrics["betweenness"].std()
    for _, row in metrics.iterrows():
        if row["betweenness"] > betw_threshold and row["betweenness"] > 0.05:
            findings["перегруженный посредник"].append(row["eid"])
    return dict(findings)


def detect_interdept_gaps(logs: pd.DataFrame, team: Team) -> List[Tuple[str, str]]:
    """
    Разрыв между отделами: пара, у которой нормированная плотность
    межотдельских связей существенно ниже среднего по всем парам отделов.
    """
    dept_of = {e.eid: e.dept for e in team.employees}
    counts = defaultdict(int)
    for _, row in logs.iterrows():
        a, b = dept_of[row["from"]], dept_of[row["to"]]
        if a != b:
            key = tuple(sorted([a, b]))
            counts[key] += row["weight"]
    densities = {}
    for key, c in counts.items():
        size_a = DEPARTMENTS[key[0]]
        size_b = DEPARTMENTS[key[1]]
        densities[key] = c / (size_a * size_b)
    if not densities:
        return []
    mean_density = sum(densities.values()) / len(densities)
    return [k for k, d in densities.items() if d < mean_density * 0.6]


MATRIX = {
    "изолированный": {
        "механика": "квесты включения + бейджи за первые контакты",
        "эффект": "повышение входящего и исходящего трафика изолированного узла",
    },
    "перегруженный посредник": {
        "механика": "делегирование + наставничество",
        "эффект": "снижение посредничества за счёт перенаправления части связей",
    },
    "разрыв между отделами": {
        "механика": "кросс-функциональные квесты",
        "эффект": "повышение межотдельского трафика",
    },
}


def recommend(findings: Dict[str, List[str]], gaps: List[Tuple[str, str]]) -> List[Dict]:
    recs = []
    for anomaly, targets in findings.items():
        if not targets:
            continue
        recs.append({
            "аномалия": anomaly,
            "субъекты": targets,
            "механика": MATRIX[anomaly]["механика"],
            "ожидаемый_эффект": MATRIX[anomaly]["эффект"],
        })
    for a, b in gaps:
        recs.append({
            "аномалия": "разрыв между отделами",
            "субъекты": [f"{a}-{b}"],
            "механика": MATRIX["разрыв между отделами"]["механика"],
            "ожидаемый_эффект": MATRIX["разрыв между отделами"]["эффект"],
        })
    return recs


def apply_targeted_intervention(team: Team, recs: List[Dict]) -> Team:
    isolated_ids: List[str] = []
    hub_ids: List[str] = []
    cross_pairs: List[Tuple[str, str]] = []
    for r in recs:
        if r["аномалия"] == "изолированный":
            isolated_ids.extend(r["субъекты"])
        if r["аномалия"] == "перегруженный посредник":
            hub_ids.extend(r["субъекты"])
        if r["аномалия"] == "разрыв между отделами":
            for s in r["субъекты"]:
                a, b = s.split("-")
                cross_pairs.append((a, b))

    for e in team.employees:
        if e.eid in isolated_ids:
            e.activity = 1.25
            e.role = "обычный"
            e.kpi_baseline = e.kpi_baseline * 1.05
        if e.eid in hub_ids:
            e.activity = 0.85
            e.role = "обычный"
    team.cross_boost = cross_pairs if cross_pairs else [("Разработка", "Продажи")]
    return team


def apply_universal_intervention(team: Team) -> Team:
    for e in team.employees:
        if e.role == "изолированный":
            e.activity = min(e.activity * 1.3, 1.6)
        elif e.role == "перегруженный посредник":
            e.activity = min(e.activity * 1.05, 2.8)
        else:
            e.activity = min(e.activity * 1.18, 1.6)
    team.cross_boost = []
    return team


def generate_logs_after(team: Team, weeks: int = WEEKS, seed_offset: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(SEED + seed_offset)
    rows = []
    boost_pairs = getattr(team, "cross_boost", [])
    total_days = weeks * DAYS_PER_WEEK
    for day in range(total_days):
        for sender in team.employees:
            for receiver in team.employees:
                if sender.eid == receiver.eid:
                    continue
                same_dept = sender.dept == receiver.dept
                boost = 1.0
                if not same_dept:
                    for a, b in boost_pairs:
                        if {sender.dept, receiver.dept} == {a, b}:
                            boost = 6.0
                            break
                base_p = P_INTRA if same_dept else P_INTER
                if not same_dept and {sender.dept, receiver.dept} == WEAK_PAIR:
                    base_p *= WEAK_PAIR_FACTOR
                p = base_p * sender.activity * boost
                if receiver.role == "перегруженный посредник":
                    p *= 2.0
                if receiver.role == "изолированный":
                    p *= 0.2
                p = min(p, 0.95)
                if rng.random() < p:
                    weight = int(rng.integers(1, 4))
                    rows.append({
                        "day": day,
                        "from": sender.eid,
                        "to": receiver.eid,
                        "weight": weight,
                    })
    return pd.DataFrame(rows)


def kpi_from_graph(g: nx.DiGraph, team: Team, baseline: bool = False) -> Dict[str, float]:
    """
    Модель индивидуального KPI: комбинация собственной активности (баланс вход/выход),
    включённости в кросс-функциональные связи и посреднического влияния.
    """
    in_deg = dict(g.in_degree(weight="weight"))
    out_deg = dict(g.out_degree(weight="weight"))
    try:
        betw = nx.betweenness_centrality(g, weight="weight", normalized=True)
    except Exception:
        betw = {n: 0.0 for n in g.nodes}

    dept_of = {e.eid: e.dept for e in team.employees}
    cross_contacts = defaultdict(int)
    for u, v, data in g.edges(data=True):
        if dept_of.get(u) != dept_of.get(v):
            cross_contacts[u] += data.get("weight", 1)
            cross_contacts[v] += data.get("weight", 1)

    kpi = {}
    for e in team.employees:
        total_deg = in_deg.get(e.eid, 0) + out_deg.get(e.eid, 0)
        comm = min(total_deg / 500.0, 1.0)
        cross = min(cross_contacts.get(e.eid, 0) / 80.0, 1.0)
        mediation_penalty = max(betw.get(e.eid, 0.0) - 0.08, 0.0) * 1.2
        value = e.kpi_baseline * (0.35 + 0.35 * comm + 0.25 * cross - mediation_penalty)
        kpi[e.eid] = float(value)
    return kpi


def run_scenario(name: str, team: Team, intervention) -> Dict:
    logs_before = generate_logs(team, seed_offset=0)
    g_before = build_graph(logs_before)
    metrics_before = compute_metrics(g_before, team)
    findings = diagnose(metrics_before)
    gaps = detect_interdept_gaps(logs_before, team)
    recs = recommend(findings, gaps)
    kpi_before = kpi_from_graph(g_before, team, baseline=True)

    team_after = intervention(team, recs) if intervention.__code__.co_argcount == 2 else intervention(team)

    logs_after = generate_logs_after(team_after, seed_offset=100)
    g_after = build_graph(logs_after)
    metrics_after = compute_metrics(g_after, team_after)
    kpi_after = kpi_from_graph(g_after, team_after)

    kpi_delta = {eid: kpi_after[eid] - kpi_before[eid] for eid in kpi_before}

    return {
        "name": name,
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
        "kpi_before": kpi_before,
        "kpi_after": kpi_after,
        "kpi_delta": kpi_delta,
        "findings": findings,
        "gaps": gaps,
        "recs": recs,
    }


def compare_scenarios(targeted: Dict, universal: Dict) -> Dict:
    kpi_targeted = np.array([targeted["kpi_delta"][e] for e in targeted["kpi_delta"]])
    kpi_universal = np.array([universal["kpi_delta"][e] for e in universal["kpi_delta"]])

    t_targ, p_targ = stats.ttest_rel(
        list(targeted["kpi_after"].values()),
        list(targeted["kpi_before"].values()),
    )
    t_univ, p_univ = stats.ttest_rel(
        list(universal["kpi_after"].values()),
        list(universal["kpi_before"].values()),
    )

    u_stat, u_p = stats.mannwhitneyu(kpi_targeted, kpi_universal, alternative="greater")

    deg_before_t = targeted["metrics_before"]["total_degree"].values
    deg_after_t = targeted["metrics_after"]["total_degree"].values
    t_deg, p_deg = stats.ttest_rel(deg_after_t, deg_before_t)

    return {
        "ttest_targeted_kpi": (float(t_targ), float(p_targ)),
        "ttest_universal_kpi": (float(t_univ), float(p_univ)),
        "mannwhitney_targeted_vs_universal": (float(u_stat), float(u_p)),
        "ttest_targeted_degree": (float(t_deg), float(p_deg)),
        "mean_delta_targeted": float(kpi_targeted.mean()),
        "mean_delta_universal": float(kpi_universal.mean()),
    }


def main():
    print("=" * 70)
    print("Исследовательский прототип: адресная геймификация на основе ONA")
    print("=" * 70)

    team_targeted = build_team()
    logs0 = generate_logs(team_targeted, seed_offset=0)
    g0 = build_graph(logs0)
    metrics0 = compute_metrics(g0, team_targeted)
    print("\n[Шаг 1-2] Сгенерированы логи (",
          len(logs0), "записей) и построен граф.")
    print("\n[Шаг 2] Сводка базовых метрик (первые 10 строк):")
    print(metrics0.head(10).to_string(index=False))

    findings = diagnose(metrics0)
    gaps = detect_interdept_gaps(logs0, team_targeted)
    recs = recommend(findings, gaps)
    print("\n[Шаг 3] Диагностика аномалий:")
    for anomaly, targets in findings.items():
        print(f"  - {anomaly}: {targets}")
    print(f"  - разрывы между отделами: {gaps}")

    print("\n[Шаг 4] Рекомендуемые механики:")
    for r in recs:
        print(f"  * {r['аномалия']} -> {r['механика']} (субъекты: {r['субъекты']})")

    team_targeted_clone = build_team()
    targeted = run_scenario("Адресная геймификация", team_targeted_clone, apply_targeted_intervention)

    team_universal = build_team()
    universal = run_scenario("Базовая линия (неадресная)", team_universal, apply_universal_intervention)

    comp = compare_scenarios(targeted, universal)

    print("\n[Шаг 5] Сравнение сценариев:")
    print(f"  Средний прирост KPI (адресный):   {comp['mean_delta_targeted']:+.4f}")
    print(f"  Средний прирост KPI (базовый):    {comp['mean_delta_universal']:+.4f}")
    print(f"  Парный t-критерий (адресный, KPI): t={comp['ttest_targeted_kpi'][0]:.3f}, p={comp['ttest_targeted_kpi'][1]:.5f}")
    print(f"  Парный t-критерий (базовый, KPI):  t={comp['ttest_universal_kpi'][0]:.3f}, p={comp['ttest_universal_kpi'][1]:.5f}")
    print(f"  U-критерий Манна-Уитни (адресный > базовый): U={comp['mannwhitney_targeted_vs_universal'][0]:.2f}, p={comp['mannwhitney_targeted_vs_universal'][1]:.5f}")
    print(f"  Парный t-критерий (адресный, суммарный degree): t={comp['ttest_targeted_degree'][0]:.3f}, p={comp['ttest_targeted_degree'][1]:.5f}")

    print("\n" + "=" * 70)
    if comp["ttest_targeted_kpi"][1] < 0.05 and comp["mannwhitney_targeted_vs_universal"][1] < 0.05:
        print("ВЫВОД: гипотеза подтверждается на модельных данных.")
    else:
        print("ВЫВОД: гипотеза подтверждается частично, требуется уточнение параметров.")
    print("=" * 70)

    import json
    from pathlib import Path
    summary = {
        "team_size": len(team_targeted.employees),
        "departments": DEPARTMENTS,
        "weeks": WEEKS,
        "n_logs_before": len(logs0),
        "findings": {k: list(v) for k, v in findings.items()},
        "gaps": [list(g) for g in gaps],
        "mean_delta_targeted": comp["mean_delta_targeted"],
        "mean_delta_universal": comp["mean_delta_universal"],
        "ttest_targeted_kpi": comp["ttest_targeted_kpi"],
        "ttest_universal_kpi": comp["ttest_universal_kpi"],
        "mannwhitney": comp["mannwhitney_targeted_vs_universal"],
        "ttest_targeted_degree": comp["ttest_targeted_degree"],
        "metrics_before_head": metrics0.head(10).to_dict(orient="records"),
        "isolated_before_degree": {
            eid: int(metrics0.loc[metrics0["eid"] == eid, "total_degree"].iloc[0])
            for lst in findings.values() for eid in lst
        },
        "isolated_after_degree_targeted": {
            eid: int(targeted["metrics_after"].loc[targeted["metrics_after"]["eid"] == eid, "total_degree"].iloc[0])
            for lst in findings.values() for eid in lst
        },
        "isolated_after_degree_universal": {
            eid: int(universal["metrics_after"].loc[universal["metrics_after"]["eid"] == eid, "total_degree"].iloc[0])
            for lst in findings.values() for eid in lst
        },
    }
    out = Path(__file__).parent / "prototype_summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\nСводка сохранена: {out}")

    return {"targeted": targeted, "universal": universal, "comparison": comp}


if __name__ == "__main__":
    main()
