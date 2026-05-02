"""
Эксперименты по чувствительности и альтернативной конфигурации.

Запускает параметризованную версию прототипа в нескольких конфигурациях:
1. Чувствительность к размеру команды: 15, 20, 25, 30, 40 человек.
2. Чувствительность к зерну генератора: SEED in {1, 7, 13, 99}
   при канонической команде.
3. Альтернативная конфигурация: 22 человека, 4 направления.

Логика порогов диагностики и формулы KPI взяты из prototype.py без изменений.
Каноническая конфигурация 25 человек воспроизводится в той же последовательности
вызовов, что и в prototype.py, что обеспечивает совпадение чисел при size_25.

Запуск: python experiments.py
Зависимости: numpy, networkx, pandas, scipy.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class ExperimentConfig:
    name: str
    departments: Dict[str, int]
    isolated_idx: List[int]
    hub_idx: List[int]
    p_intra: float = 0.45
    p_inter: float = 0.03
    isolated_factor: float = 0.1
    hub_factor: float = 2.8
    weak_pair: Optional[Tuple[str, str]] = None
    weak_pair_factor: float = 0.25
    seed: int = 42
    weeks: int = 4
    days_per_week: int = 5


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


def build_team(cfg: ExperimentConfig) -> Team:
    employees: List[Employee] = []
    counter = 1
    for dept, n in cfg.departments.items():
        for _ in range(n):
            eid = f"E{counter:02d}"
            activity = float(np.clip(np.random.normal(1.0, 0.2), 0.4, 1.6))
            employees.append(Employee(eid=eid, dept=dept, activity=activity))
            counter += 1

    for idx in cfg.isolated_idx:
        if 0 <= idx < len(employees):
            employees[idx].role = "изолированный"
            employees[idx].activity = cfg.isolated_factor

    for i, idx in enumerate(cfg.hub_idx):
        if 0 <= idx < len(employees):
            employees[idx].role = "перегруженный посредник"
            employees[idx].activity = cfg.hub_factor * (1.0 if i == 0 else 0.9)

    for e in employees:
        e.kpi_baseline = float(np.clip(np.random.normal(1.0, 0.15), 0.6, 1.4))
    return Team(employees)


def generate_logs(team: Team, cfg: ExperimentConfig, after: bool = False,
                  cross_boost: Optional[List[Tuple[str, str]]] = None) -> pd.DataFrame:
    seed_offset = 100 if after else 0
    rng = np.random.default_rng(cfg.seed + seed_offset)
    cross_boost = cross_boost or []
    rows = []
    total_days = cfg.weeks * cfg.days_per_week
    for day in range(total_days):
        for sender in team.employees:
            for receiver in team.employees:
                if sender.eid == receiver.eid:
                    continue
                same_dept = sender.dept == receiver.dept
                base_p = cfg.p_intra if same_dept else cfg.p_inter
                if (not same_dept and cfg.weak_pair
                        and {sender.dept, receiver.dept} == set(cfg.weak_pair)):
                    base_p *= cfg.weak_pair_factor
                boost = 1.0
                if not same_dept:
                    for a, b in cross_boost:
                        if {sender.dept, receiver.dept} == {a, b}:
                            boost = 6.0
                            break
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


def build_graph(logs: pd.DataFrame) -> nx.DiGraph:
    g = nx.DiGraph()
    if len(logs) == 0:
        return g
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
    if metrics["betweenness"].std() > 0:
        betw_threshold = (metrics["betweenness"].mean()
                          + 1.5 * metrics["betweenness"].std())
    else:
        betw_threshold = float("inf")
    for _, row in metrics.iterrows():
        if row["betweenness"] > betw_threshold and row["betweenness"] > 0.05:
            findings["перегруженный посредник"].append(row["eid"])
    return dict(findings)


def detect_interdept_gaps(logs: pd.DataFrame, team: Team,
                          cfg: ExperimentConfig) -> List[Tuple[str, str]]:
    dept_of = {e.eid: e.dept for e in team.employees}
    counts = defaultdict(int)
    for _, row in logs.iterrows():
        a, b = dept_of[row["from"]], dept_of[row["to"]]
        if a != b:
            key = tuple(sorted([a, b]))
            counts[key] += row["weight"]
    densities = {}
    for key, c in counts.items():
        size_a = cfg.departments[key[0]]
        size_b = cfg.departments[key[1]]
        densities[key] = c / (size_a * size_b)
    if not densities:
        return []
    mean_density = sum(densities.values()) / len(densities)
    return [k for k, d in densities.items() if d < mean_density * 0.6]


def recommend(findings: Dict[str, List[str]],
              gaps: List[Tuple[str, str]]) -> List[Dict]:
    recs = []
    for anomaly, targets in findings.items():
        if not targets:
            continue
        recs.append({"аномалия": anomaly, "субъекты": targets})
    for a, b in gaps:
        recs.append({"аномалия": "разрыв между отделами",
                     "субъекты": [f"{a}-{b}"]})
    return recs


def apply_targeted_intervention(team: Team, recs: List[Dict]
                                ) -> Tuple[Team, List[Tuple[str, str]]]:
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
    return team, cross_pairs


def apply_universal_intervention(team: Team) -> Team:
    for e in team.employees:
        if e.role == "изолированный":
            e.activity = min(e.activity * 1.3, 1.6)
        elif e.role == "перегруженный посредник":
            e.activity = min(e.activity * 1.05, 2.8)
        else:
            e.activity = min(e.activity * 1.18, 1.6)
    return team


def kpi_from_graph(g: nx.DiGraph, team: Team) -> Dict[str, float]:
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
        value = e.kpi_baseline * (0.35 + 0.35 * comm + 0.25 * cross
                                  - mediation_penalty)
        kpi[e.eid] = float(value)
    return kpi


def run_full_experiment(cfg: ExperimentConfig) -> Dict:
    """
    Запускает оба сценария (адресный и базовый) и возвращает сводку.

    Воспроизводит последовательность вызовов из prototype.main():
    три последовательных build_team и две генерации логов на каждом сценарии.
    """
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Шаг 0: команда для предварительной диагностики (как в prototype.main()).
    team_diag = build_team(cfg)
    logs_diag = generate_logs(team_diag, cfg, after=False)
    g_diag = build_graph(logs_diag)
    metrics_diag = compute_metrics(g_diag, team_diag)
    findings_diag = diagnose(metrics_diag)
    gaps_diag = detect_interdept_gaps(logs_diag, team_diag, cfg)

    # Адресный сценарий.
    team_t = build_team(cfg)
    logs_before_t = generate_logs(team_t, cfg, after=False)
    g_before_t = build_graph(logs_before_t)
    metrics_before_t = compute_metrics(g_before_t, team_t)
    findings_t = diagnose(metrics_before_t)
    gaps_t = detect_interdept_gaps(logs_before_t, team_t, cfg)
    recs_t = recommend(findings_t, gaps_t)
    kpi_before_t = kpi_from_graph(g_before_t, team_t)
    team_t, cross_pairs = apply_targeted_intervention(team_t, recs_t)
    if not cross_pairs:
        depts = list(cfg.departments.keys())
        if len(depts) >= 2:
            cross_pairs = [(depts[0], depts[1])]
    logs_after_t = generate_logs(team_t, cfg, after=True, cross_boost=cross_pairs)
    g_after_t = build_graph(logs_after_t)
    metrics_after_t = compute_metrics(g_after_t, team_t)
    kpi_after_t = kpi_from_graph(g_after_t, team_t)

    # Базовый сценарий.
    team_u = build_team(cfg)
    logs_before_u = generate_logs(team_u, cfg, after=False)
    g_before_u = build_graph(logs_before_u)
    metrics_before_u = compute_metrics(g_before_u, team_u)
    kpi_before_u = kpi_from_graph(g_before_u, team_u)
    team_u = apply_universal_intervention(team_u)
    logs_after_u = generate_logs(team_u, cfg, after=True, cross_boost=[])
    g_after_u = build_graph(logs_after_u)
    metrics_after_u = compute_metrics(g_after_u, team_u)
    kpi_after_u = kpi_from_graph(g_after_u, team_u)

    # Статистика.
    kpi_delta_t = np.array([kpi_after_t[e] - kpi_before_t[e] for e in kpi_before_t])
    kpi_delta_u = np.array([kpi_after_u[e] - kpi_before_u[e] for e in kpi_before_u])
    t_targ, p_targ = stats.ttest_rel(list(kpi_after_t.values()),
                                     list(kpi_before_t.values()))
    t_univ, p_univ = stats.ttest_rel(list(kpi_after_u.values()),
                                     list(kpi_before_u.values()))
    u_stat, u_p = stats.mannwhitneyu(kpi_delta_t, kpi_delta_u, alternative="greater")
    deg_before_t_arr = metrics_before_t["total_degree"].values
    deg_after_t_arr = metrics_after_t["total_degree"].values
    t_deg, p_deg = stats.ttest_rel(deg_after_t_arr, deg_before_t_arr)

    return {
        "name": cfg.name,
        "team_size": sum(cfg.departments.values()),
        "departments": dict(cfg.departments),
        "seed": cfg.seed,
        "p_intra": cfg.p_intra,
        "p_inter": cfg.p_inter,
        "mean_delta_targeted": float(kpi_delta_t.mean()),
        "mean_delta_universal": float(kpi_delta_u.mean()),
        "ttest_targeted_kpi": (float(t_targ), float(p_targ)),
        "ttest_universal_kpi": (float(t_univ), float(p_univ)),
        "mannwhitney": (float(u_stat), float(u_p)),
        "ttest_targeted_degree": (float(t_deg), float(p_deg)),
        "findings_targeted": {k: list(v) for k, v in findings_t.items()},
        "gaps_targeted": [list(g) for g in gaps_t],
    }


def make_cfg(name, departments, seed=42, p_intra=0.45, p_inter=0.03,
             isolated_factor=0.1, hub_factor=2.8,
             isolated_idx=None, hub_idx=None, weak_pair=None):
    sizes = list(departments.values())
    cum = [0]
    for s in sizes:
        cum.append(cum[-1] + s)
    if isolated_idx is None:
        isolated_idx = [cum[0] + min(2, sizes[0] - 1)]
        if len(sizes) >= 2:
            isolated_idx.append(cum[1] + min(4, sizes[1] - 1))
    if hub_idx is None:
        hub_idx = [0]
        if len(sizes) >= 2:
            hub_idx.append(cum[1])
    return ExperimentConfig(
        name=name,
        departments=departments,
        isolated_idx=isolated_idx,
        hub_idx=hub_idx,
        p_intra=p_intra,
        p_inter=p_inter,
        isolated_factor=isolated_factor,
        hub_factor=hub_factor,
        weak_pair=weak_pair,
        seed=seed,
    )


def run_sensitivity_size():
    cfgs = [
        make_cfg("size_15", {"Разработка": 6, "Продажи": 5, "Поддержка": 4},
                 weak_pair=("Разработка", "Продажи")),
        make_cfg("size_20", {"Разработка": 8, "Продажи": 7, "Поддержка": 5},
                 weak_pair=("Разработка", "Продажи")),
        make_cfg("size_25", {"Разработка": 10, "Продажи": 8, "Поддержка": 7},
                 weak_pair=("Разработка", "Продажи")),
        make_cfg("size_30", {"Разработка": 12, "Продажи": 10, "Поддержка": 8},
                 weak_pair=("Разработка", "Продажи")),
        make_cfg("size_40", {"Разработка": 16, "Продажи": 13, "Поддержка": 11},
                 weak_pair=("Разработка", "Продажи")),
    ]
    return [run_full_experiment(c) for c in cfgs]


def run_sensitivity_seed():
    out = []
    for s in [1, 7, 13, 99]:
        cfg = make_cfg(f"seed_{s}",
                       {"Разработка": 10, "Продажи": 8, "Поддержка": 7},
                       seed=s, weak_pair=("Разработка", "Продажи"))
        out.append(run_full_experiment(cfg))
    return out


def run_sensitivity_p():
    """Чувствительность к коммуникационным вероятностям при канонической команде."""
    out = []
    grid = [
        ("p_intra_low", 0.36, 0.03),     # -20%
        ("p_intra_high", 0.54, 0.03),    # +20%
        ("p_inter_low", 0.45, 0.024),    # -20%
        ("p_inter_high", 0.45, 0.036),   # +20%
    ]
    for name, p_in, p_out in grid:
        cfg = make_cfg(name, {"Разработка": 10, "Продажи": 8, "Поддержка": 7},
                       p_intra=p_in, p_inter=p_out,
                       weak_pair=("Разработка", "Продажи"))
        out.append(run_full_experiment(cfg))
    return out


def run_alternative_config():
    """
    Альтернативная конфигурация: 22 человека в 4 направлениях.
    Состав: Оргкомитет 8, SMM_PR 5, Дизайн 5, Руководство 4.
    Хабы (перегруженные посредники): первый из Руководства (индекс 18) и
        запасной хаб - глава Оргкомитета (индекс 0).
    Изоляты: новички в Оргкомитете (индекс 3) и SMM_PR (индекс 11).
    Слабая межотдельная пара: Оргкомитет - Дизайн.
    """
    departments = {
        "Оргкомитет": 8,
        "SMM_PR": 5,
        "Дизайн": 5,
        "Руководство": 4,
    }
    cfg = ExperimentConfig(
        name="alternative_config",
        departments=departments,
        isolated_idx=[3, 11],
        hub_idx=[18, 0],
        weak_pair=("Оргкомитет", "Дизайн"),
    )
    return run_full_experiment(cfg)


def main():
    print("=" * 70)
    print("Эксперименты по чувствительности и альтернативной конфигурации")
    print("=" * 70)

    results = {
        "sensitivity_size": run_sensitivity_size(),
        "sensitivity_seed": run_sensitivity_seed(),
        "sensitivity_p": run_sensitivity_p(),
        "alternative_config": run_alternative_config(),
    }

    print("\n--- Чувствительность к размеру команды ---")
    for r in results["sensitivity_size"]:
        print(f"  {r['name']:>10}: n={r['team_size']:>2}, "
              f"d_targ={r['mean_delta_targeted']:+.4f}, "
              f"d_univ={r['mean_delta_universal']:+.4f}, "
              f"p(U)={r['mannwhitney'][1]:.5f}, "
              f"p(t_deg)={r['ttest_targeted_degree'][1]:.5f}")

    print("\n--- Чувствительность к зерну ---")
    for r in results["sensitivity_seed"]:
        print(f"  {r['name']:>10}: d_targ={r['mean_delta_targeted']:+.4f}, "
              f"d_univ={r['mean_delta_universal']:+.4f}, "
              f"p(U)={r['mannwhitney'][1]:.5f}")

    print("\n--- Чувствительность к p_intra / p_inter ---")
    for r in results["sensitivity_p"]:
        print(f"  {r['name']:>14}: p_intra={r['p_intra']:.3f}, "
              f"p_inter={r['p_inter']:.3f}, "
              f"d_targ={r['mean_delta_targeted']:+.4f}, "
              f"d_univ={r['mean_delta_universal']:+.4f}, "
              f"p(U)={r['mannwhitney'][1]:.5f}")

    print("\n--- Альтернативная конфигурация (4 отдела, 22 человека) ---")
    r = results["alternative_config"]
    print(f"  Команда: {r['team_size']} человек, "
          f"{len(r['departments'])} направления: {r['departments']}")
    print(f"  d_targ={r['mean_delta_targeted']:+.4f}, "
          f"d_univ={r['mean_delta_universal']:+.4f}")
    print(f"  t (адресный, KPI): t={r['ttest_targeted_kpi'][0]:.3f}, "
          f"p={r['ttest_targeted_kpi'][1]:.5f}")
    print(f"  U (адресный > базовый): U={r['mannwhitney'][0]:.2f}, "
          f"p={r['mannwhitney'][1]:.5f}")
    print(f"  t (адресный, degree): t={r['ttest_targeted_degree'][0]:.3f}, "
          f"p={r['ttest_targeted_degree'][1]:.5f}")
    print(f"  Найденные аномалии: {r['findings_targeted']}")
    print(f"  Найденные межотдельные разрывы: {r['gaps_targeted']}")

    out = Path(__file__).parent / "experiments_summary.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str),
                   encoding="utf-8")
    print(f"\nСводка сохранена: {out}")


if __name__ == "__main__":
    main()
