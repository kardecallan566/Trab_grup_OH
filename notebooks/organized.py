# TODO ver commit anterior e mudar aqui

# imports
import itertools
import pandas as pd
import random
import plotnine.options
from plotnine import *
from pprint import pprint as pp

plotnine.options.figure_size = (7, 5)


# region functions utils
def first(iterable, func, default=None):
  # get first element that satisfies func
  return next(filter(func, iterable), default)


def first_ind(iterable, func, default=None):
  # get index of first element that satisfies func
  return next((i for i, x in enumerate(iterable) if func(x)), default)


set_seed = random.seed
# endregion

# region types
horas = int
tarefa = int
periodo = (horas, horas)
cromossoma = list[tarefa]
populacaoT = list[cromossoma]


class Maquina:
  def __init__(self, numero: int, n_tarefas: int):
    self.numero = numero
    self.n_tarefas = n_tarefas

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


def cromo_to_horario(crom: cromossoma) -> dict[tarefa, periodo]:
  maqs: dict[Maquina, list[tarefa]] = cromo_to_maqs(crom)
  horario: dict[Maquina, list[periodo]] = maqs_to_horario(maqs)
  ret = {}
  for m, l in horario.items():
    tarefa_periodo: tuple[tarefa, periodo] = zip(maqs[m], l)
    for t, p in tarefa_periodo:
      ret[t] = p
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
    scale_x_continuous(breaks=[0, 24, 48, 72, 96, max_tempo]) + \
    scale_y_continuous(breaks=[1, 2, 3]) + \
    scale_color_discrete(guide=False) + \
    coord_fixed(xlim=(-0.5, 144)) + \
    labs(x="horas", y="maquina") + \
    theme_classic() + \
    theme(aspect_ratio=0.7)


def cromo_respeita_restricao(cromo: cromossoma) -> bool:
  horario: dict[tarefa, periodo] = cromo_to_horario(cromo)
  for t, p in horario.items():
    if t in restricao and p[1] > restricao[t]:
      return False
  return True


def get_tempo_total(cromo: cromossoma) -> horas:
  # tempo máximo de todas as máquinas
  return max([p[1] for t, p in cromo_to_horario(cromo).items()])


def cromos_to_df(*cromos: cromossoma) -> pd.DataFrame:
  # hardcoded, supposed to be:
  # M1: tarefas do M1,
  # M2: tarefas do M2,
  # M3: tarefas do M3,
  # tempo total: tempo máximo de todas as máquinas
  # respeita restricao?: se o horario das tarefas com restricoes acaba antes do valor da restricao
  # respeita unique?: se o cromo nao tem tarefas repetidas
  return pd.DataFrame({
    "M1": [" -> ".join([f"T{i}" for i in c[:4]]) for c in cromos],
    "M2": [" -> ".join([f"T{i}" for i in c[4:7]]) for c in cromos],
    "M3": [" -> ".join([f"T{i}" for i in c[7:]]) for c in cromos],
    "tempo total": [get_tempo_total(c) for c in cromos],
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
  return ggplot(df, aes(x="factor(tarefa)", y="tempo")) + \
    geom_bar(aes(fill="factor(prioridade)"), stat="identity") + \
    geom_text(aes(label="tempo"), nudge_y=1) + \
    scale_fill_discrete(name="Com restrição?") + \
    labs(x="tarefa", y="tempo") + \
    theme_classic()


def get_maquina(index: int) -> Maquina:
  return maquinas[index - 1]


def tarefas_as_list() -> list[int]:
  return list(Tarefas.keys())

def maqs_to_cromo(maqs: dict[Maquina, list[tarefa]]) -> cromossoma:
  return [t for m in maqs for t in maqs[m]]

def from_ina_to_adm(cromo: cromossoma) -> cromossoma:
  # puxar pra esquerda todas as tarefas de prioridade, n ve por unicos
  maqs: dict[Maquina, list[tarefa]] = cromo_to_maqs(cromo)
  tarefas_com_restricao = list(restricao.keys())
  for maq in maqs:
    maqs[maq] = sorted(maqs[maq], key=lambda t: t in tarefas_com_restricao, reverse=True)
  return maqs_to_cromo(maqs)

# endregion

# region funcoes algoritmo
def heuristica() -> cromossoma:
  ret = [0] * len(Tarefas)
  tarefas = tarefas_as_list()
  M1 = get_maquina(1)
  M2 = get_maquina(2)
  M3 = get_maquina(3)
  M1_ind = 0
  M2_ind = M1_ind + M1.n_tarefas
  M3_ind = M2_ind + M2.n_tarefas
  # 1 - atribuir as 3 tarefas criticas às 3 máquinas, M1 recebendo a mais pequena sendo que tem 4
  criticas = list(restricao.keys())
  criticas.sort(key=lambda t: get_time_of_tarefa(t), reverse=False)
  ret[M1_ind], ret[M2_ind], ret[M3_ind] = criticas
  [tarefas.remove(c) for c in criticas]
  # 2 - como a M1 tem 4 tarefas, atribuir as 3 tarefas mais pequenas à M1
  tarefas.sort(key=lambda t: get_time_of_tarefa(t), reverse=False)
  ret[M1_ind + 1:M1_ind + 4] = tarefas[:3]
  # 3 - dividir as restantes tarefas pelas máquinas de forma equilibrada
  # ou seja, M2 vai ter as tarefas mais pequena e maior (das restantes) e M3 vai ter as tarefas do meio
  ret[M2_ind + 1], ret[M3_ind + 1], ret[M3_ind + 2], ret[M2_ind + 2] = tarefas[3:]
  return ret


# ERO
def crossover(pai: list[int], mae: list[int], log=False, seed=None) -> list[int]:
  if seed is not None: set_seed(seed)
  # selecionar aleatoriamente uma ligacao entre 2 genes, entre os pais, e adicionar a um novo filho
  filho = []
  nao_escolhidos = set(pai)

  # 1º escolher um ponto de partida aleatorio
  ponto_partida = escolhido = random.choice([pai[0], mae[0]])
  # 2º fazer as matrizes de adjacencia (incluindo os ultimos)
  pai_adj = {pai[i]: {pai[i - 1], pai[(i + 1) % len(pai)]} for i in range(len(pai))}
  mae_adj = {mae[i]: {mae[i - 1], mae[(i + 1) % len(mae)]} for i in range(len(mae))}
  # 3º fazer uniao das matrizes de adjacencia
  adj = {i: pai_adj[i] | mae_adj[i] for i in pai_adj}
  if log:
    print("Adjacencias:")
    pp(adj)
  # 4º fazer o filho
  while len(nao_escolhidos) > 1:
    if log: print("Não escolhidos:", nao_escolhidos)
    if log: print("Filho:", filho)
    if log: print("Escolhido:", escolhido)
    # adicionar o escolhido ao filho
    filho.append(escolhido)
    nao_escolhidos.remove(escolhido)

    # conseguir as escolhas possiveis
    escolhas = adj[escolhido] & nao_escolhidos
    if len(escolhas) == 0:
      # se nao houver escolhas, escolher um aleatorio entre os nao escolhidos
      escolhido = random.choice(list(nao_escolhidos))
    else:
      escolhido = random.choice(list(escolhas))
  # adicionar o filho que falta
  filho.append(nao_escolhidos.pop())
  return filho


cruzamento = crossover


def mutacao(crom: cromossoma, prob=.1, seed=None):
  if seed is not None: set_seed(seed)
  crom = crom.copy()
  # Verificar se a mutação será aplicada com base na probabilidade
  if random.random() < prob:
    # Selecionar aleatoriamente duas posições diferentes no cromossomo
    ind1 = random.randint(0, len(crom) - 1)
    ind2 = random.randint(0, len(crom) - 1)

    # Realizar a troca entre as duas posições
    crom[ind1], crom[ind2] = crom[ind2], crom[ind1]
  return crom


def aptidao(crom: cromossoma, penalizao=20) -> int:
  # A aptidao equivale ao tempo total de execução do cromossomo, mais penalizacao
  rest = cromo_respeita_restricao(crom)
  return get_tempo_total(crom) + (0 if rest else penalizao)


def cromos_to_df_updated(*cromos: cromossoma) -> pd.DataFrame:
  # same as the one above but sem respeita unique (q é garantido) e com apitdao
  return pd.DataFrame({
    "M1": [" -> ".join([f"T{i}" for i in c[:4]]) for c in cromos],
    "M2": [" -> ".join([f"T{i}" for i in c[4:7]]) for c in cromos],
    "M3": [" -> ".join([f"T{i}" for i in c[7:]]) for c in cromos],
    "tempo total": [get_tempo_total(c) for c in cromos],
    "respeita restricao?": [cromo_respeita_restricao(c) for c in cromos],
    "aptidao": [aptidao(c) for c in cromos]
  })


class CriteriosDeParagem:
  def __init__(self, iter_max=None, no_improv_max=None, aptidao_menos_que=None):
    self.iter_max: int = iter_max
    self.no_improv_max: int = no_improv_max
    self.aptidao_menos_que: int = aptidao_menos_que

    self.curr_iter = 0
    self.no_improv_iter = 0
    self.current_aptid = 0

  def shouldStop(self) -> bool:
    return (
        (self.iter_max is not None and self.curr_iter >= self.iter_max) or
        (self.no_improv_max is not None and self.no_improv_iter >= self.no_improv_max) or
        (self.aptidao_menos_que is not None and self.current_aptid >= self.aptidao_menos_que)
    )

  def update(self, max_aptid_found: int):
    self.curr_iter += 1
    self.no_improv_iter += 1 if max_aptid_found > self.current_aptid else 0
    self.current_aptid = max(self.current_aptid, max_aptid_found)


# selecao proporcional à aptidao
def selecao(populacao: populacaoT) -> tuple[cromossoma, cromossoma]:
  # calcular a aptidao comulativa
  sumcum = list(itertools.accumulate([aptidao(c) for c in populacao]))
  # escolher um numero aleatorio entre 0 e a soma da aptidao
  r = random.randint(0, sumcum[-1])
  # encontrar o indice do primeiro elemento maior que o numero aleatorio
  ind1 = first_ind(sumcum, lambda x: x > r)
  # 2 vezes
  r = random.randint(0, sumcum[-1])
  ind2 = first_ind(sumcum, lambda x: x > r)

  return populacao[ind1], populacao[ind2]


def populacaoInicial(tamanho: int, tamanho_cromo=10, log=False, seed=None) -> populacaoT:
  if seed is not None: set_seed(seed)
  populacao = []
  for i in range(tamanho):
    cromo = random.sample(range(1, tamanho_cromo + 1), tamanho_cromo)
    if log: print(f"Cromo {i + 1}: {cromo}")
    populacao.append(cromo)
  return populacao


def algoritmo_genetico(
    tamanho_populacao: int,
    criterios_de_paragem: CriteriosDeParagem = CriteriosDeParagem(),
    elitismo_tamanho: int = 1,
    log=False
) -> tuple[cromossoma, list[populacaoT]]:
  ret = []
  # 1. Gerar a população inicial
  populacao = sorted(populacaoInicial(tamanho_populacao, log=log), key=lambda c: aptidao(c))
  elites: list[cromossoma] = [populacao[i] for i in range(elitismo_tamanho)]
  # 2. popular uma nova população
  while not criterios_de_paragem.shouldStop():
    if log: print(f"iteracao {criterios_de_paragem.curr_iter}")
    new_populacao = []
    for _ in range(tamanho_populacao):  # modelo geracional
      # 2.1 seleção dos pais
      pai, mae = selecao(populacao)
      # 2.2 cruzamento
      filho = cruzamento(pai, mae)
      # 2.3 mutação
      filho = mutacao(filho)
      # 2.4 adicionar o filho à nova população
      new_populacao.append(filho)
    # 3. elitismo
    # 3.1 obter o(s) menos abto(s)
    new_populacao = sorted(new_populacao, key=lambda c: aptidao(c), reverse=False)
    # 3.2 substituir o(s) menos apto(s) pelo(s) mais apto(s) da população anterior
    for i in range(elitismo_tamanho):
      new_populacao[i] = elites[i]
    # 3.3 atualizar os elites se necessário
    [elites.append(new_populacao[i]) for i in range(elitismo_tamanho)]
    elites = sorted(elites, key=lambda c: aptidao(c), reverse=False)[:elitismo_tamanho]
    # 4. atualizar a população
    ret.append(populacao := new_populacao)
    # 5. critérios de paragem
    criterios_de_paragem.update(aptidao(elites[0]))
  return elites[0], ret


# endregion


alineas = {
  "a)": plot_tarefas,
  "b)": heuristica,
  "c)": heuristica,
  "d)": cruzamento,
  "e)": mutacao,
  "f)": aptidao,
  "g)": algoritmo_genetico,
}
