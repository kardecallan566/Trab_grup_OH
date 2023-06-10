# imports
from dataclasses import dataclass as dc
import itertools
import pandas as pd
import numpy as np
import random
from dataclasses import dataclass as dc
from typing import Iterable
from collections import namedtuple
from plotnine import *
from copy import deepcopy


# region functions utils
def first(iterable, func, default=None):
  # get first element that satisfies func
  return next(filter(func, iterable), default)


def first_ind(iterable, func, default=None):
  # get index of first element that satisfies func
  return next((i for i, x in enumerate(iterable) if func(x)), default)


seed = random.seed
# endregion

# region types
horas = int
tarefa = int
periodo = (horas, horas)
cromossoma = list[tarefa]

@dc
class Maquina:
  numero: int
  n_tarefas: int

  def __repr__(self):
    return f'M{self.numero}'

  def __hash__(self):
    return self.numero


# endregion

# region values

Tarefas: dict[tarefa, horas] = {
  t + 1: h for t, h in enumerate([38, 33, 36, 20, 32, 29, 46, 38, 34, 40])
}
restricao_val = acabarAntes = 24 * (5 - 1)
restricao: dict[tarefa, horas] = {  # tempo minimo para a compleção das tarefa
  3: restricao_val,
  4: restricao_val,
  5: restricao_val,
}
maquinas = [Maquina(1, 4), Maquina(2, 3), Maquina(3, 3)]

# endregion

# region functions object like
def get_time_of_tarefa(t: tarefa) -> horas:
  return Tarefas[t]

def get_tarefa_from_cromo(cromo: cromossoma, t: tarefa) -> tarefa:
  return cromo[t - 1]

def cromo_to_maqs(cromo: cromossoma) -> dict[Maquina, list[tarefa]]:
  # convert chromosome to machine dictionary
  ret = {m: [] for m in maquinas}
  cut = 0
  for maq, l in ret.items():
    l.extend(cromo[cut:cut + maq.n_tarefas])
    cut += maq.n_tarefas
  return ret


def maqs_to_horario(maqs: dict[Maquina, list[tarefa]]) -> dict[Maquina, list[periodo]]:
  # horario sao as horas acomuladas
  ret = {m: [] for m in maquinas}
  for maq, l in maqs.items():
    ret[maq].append((0, get_time_of_tarefa(l[0])))
    for t in l[1:]:
      ret[maq].append((ret[maq][-1][1], ret[maq][-1][1] + get_time_of_tarefa(t)))
  return ret


def plot_cromo(cromo: cromossoma):
  maqs = cromo_to_maqs(cromo)
  horario = maqs_to_horario(maqs)
  max_tempo = max([i[1] for m in horario for i in horario[m]])
  df = pd.DataFrame({
    'maquina': [i for m in maqs for i in [m.numero] * len(maqs[m])],
    'tarefa': [t for m in maqs for t in maqs[m]],
    "tarefaLabel": [f"T{t}" for m in maqs for t in maqs[m]],
    "comeca": [i[0] for m in horario for i in horario[m]],
    "acaba": [i[1] for m in horario for i in horario[m]],
  })
  # noinspection PyTypeChecker,DuplicatedCode
  return ggplot(df, aes(y="maquina")) + geom_point(aes(x="comeca", color="factor(comeca)"), size=5) + \
    geom_path(aes(x="acaba", group="maquina", color="factor(acaba)")) + \
    geom_path(aes(x="comeca", group="maquina", color="factor(comeca)")) + \
    geom_vline(xintercept=max_tempo, linetype="dashed") + \
    geom_text(aes(x="comeca", label="tarefaLabel"), size=10, nudge_x=0.1, nudge_y=0.1) + \
 \
    scale_x_continuous(breaks=[0, 24, 48, 72, 96, 120, max_tempo]) + \
    scale_y_continuous(breaks=[1, 2, 3]) + \
    scale_color_discrete(guide=False) + \
    coord_fixed(xlim=(-0.5, 144)) + \
    labs(x="horas", y="maquina") + \
    theme_classic() + \
    theme(aspect_ratio=0.7)


def cromo_respeita_restricao(cromo: cromossoma) -> bool:
  # se o horario das tarefas com restricoes acaba antes do valor da restricao
  return all(
    maqs_to_horario(cromo_to_maqs(cromo))[m][-1][1] <= restricao[m.numero]
    for m in maquinas if m.numero in restricao
  )


def cromos_to_df(*cromos: list[cromossoma]) -> pd.DataFrame:
  # hardcoded, supposed to be:
  # M1: tarefas do M1,
  # M2: tarefas do M2,
  # M3: tarefas do M3,
  # tempo total: tempo máximo de todas as máquinas
  # respeita restricao?: se o horario das tarefas com restricoes acaba antes do valor da restricao
  # respeita unique?: se o cromo nao tem tarefas repetidas
  return pd.DataFrame({
    "M1": [c[:4] for c in cromos],
    "M2": [c[4:7] for c in cromos],
    "M3": [c[7:] for c in cromos],
    "tempo total": [max(maqs_to_horario(cromo_to_maqs(c))[m][-1][1] for m in maquinas) for c in cromos],
    "respeita restricao?": [cromo_respeita_restricao(c) for c in cromos],
    "respeita unique?": [len(c) == len(set(c)) for c in cromos],
  })


def plot_tarefas():
  # bar plot com as tarefas e o tempo delas
  df = pd.DataFrame({
    "tarefa": [t for t in Tarefas],
    "tempo": [get_time_of_tarefa(t) for t in Tarefas],
    "prioridade": [t in restricao for t in Tarefas]
  })
  # noinspection PyTypeChecker,DuplicatedCode
  ggplot(df, aes(x="factor(tarefa)", y="tempo")) + \
    geom_bar(aes(fill="factor(prioridade)"), stat="identity") + \
    geom_text(aes(label="tempo"), nudge_y=1) + \
    scale_fill_discrete(name="Com restrição?") + \
    labs(x="tarefa", y="tempo") + \
    theme_classic()


# endregion

# region funcoes algoritmo
def heuristica() -> cromossoma:
  #TODO
  pass

# endregion

alineas = {
  "a)": plot_tarefas,
}