import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PASTA = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
CSV, DB = PASTA / "Atividade2.csv", PASTA / "saude_avc.db"
MOSTRAR_GRAFICOS = False

FAIXAS = [0, 20, 40, 50, 60, 70, 80, 120]
ROTULOS = ["0-19", "20-39", "40-49", "50-59", "60-69", "70-79", "80+"]
TABAGISMO = {"never smoked": "Nunca fumou", "formerly smoked": "Ex-fumante",
             "smokes": "Fumante", "Unknown": "Desconhecido"}
COR_SEM, COR_COM = "#4C72B0", "#C44E52"
pd.set_option("display.width", 120)


def secao(texto, principal=False):
    print(f"\n{'=' * 75}\n{texto}\n{'=' * 75}" if principal else f"\n--- {texto}")


def br(valor, casas=1):
    return f"{valor:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def mostrar(tabela, casas=1, **opcoes):
    formatos = {}
    for coluna in tabela.select_dtypes("float").columns:
        k = casas
        if k is None:
            valores = tabela[coluna].dropna()
            k = next(c for c in (0, 1, 2) if (valores.round(c) == valores.round(2)).all())
        formatos[coluna] = lambda v, k=k: "NaN" if pd.isna(v) else br(v, k)
    print(tabela.to_string(formatters=formatos, **opcoes))


NOMES = {"casos_avc": "casos de AVC", "taxa_avc_pct": "taxa de AVC (%)",
         "idade_media": "idade média"}


def carregar_e_tratar():
    secao("1. PREPARAÇÃO E LIMPEZA DOS DADOS", True)
    df = pd.read_csv(CSV, na_values=["N/A"])

    secao("1.1 Visão geral da base")
    print(f"Registros: {len(df)} | Características (colunas): {df.shape[1]}\n")
    print(df.dtypes.to_string(), "\n")
    mostrar(df.head(), casas=2)
    print()
    mostrar(df.drop(columns="id").describe(), casas=2)

    secao("1.2 Diagnóstico: ausentes e inconsistências")
    ausentes = df.isna().sum()
    com_avc = df["stroke"] == 1
    print("Ausentes por coluna:", ausentes[ausentes > 0].to_dict())
    imc_avc = df.loc[com_avc, "bmi"].isna().sum()
    print(f"IMC ausente entre pacientes com AVC: {imc_avc} de {com_avc.sum()} "
          f"({br(100 * imc_avc / com_avc.sum())}%)")
    print(f"Linhas duplicadas: {df.duplicated().sum()} | "
          f"IDs repetidos: {df['id'].duplicated().sum()}")
    print("Gênero:", df["gender"].value_counts().to_dict())
    print("Tabagismo:", df["smoking_status"].value_counts().to_dict())

    secao("1.3 Tratamentos aplicados")
    linhas_antes = len(df)
    df = df.rename(columns={"Residence_type": "residence_type"})
    df = df[df["gender"] != "Other"].copy()
    df["faixa_etaria"] = pd.cut(df["age"], FAIXAS, labels=ROTULOS, right=False)

    df["bmi_imputado"] = df["bmi"].isna().astype(int)
    por_faixa = df.groupby("faixa_etaria", observed=True)["bmi"]
    print("(a) 'Residence_type' renomeada para 'residence_type'")
    print("(b) Removido 1 registro com gênero 'Other'")
    medianas = " | ".join(f"{faixa}: {br(m)}" for faixa, m in por_faixa.median().items())
    print("(c) IMC ausente preenchido com a mediana da faixa etária:", medianas)
    print("(d) Tabagismo 'Unknown' mantido como categoria (removê-lo apagaria 30% da base)")
    df["bmi"] = df["bmi"].fillna(por_faixa.transform("median"))

    secao("1.4 Evidências de que os dados foram tratados")
    print(f"Registros: {linhas_antes} -> {len(df)}")
    print(f"Valores ausentes: {ausentes.sum()} -> {df.isna().sum().sum()}\n")
    print(df.isna().sum().to_string())
    print("\nExemplos de registros com IMC preenchido:")
    exemplos = df.loc[df["bmi_imputado"] == 1, ["id", "age", "faixa_etaria", "bmi", "stroke"]]
    mostrar(exemplos.head(6), index=False)
    return df


def analisar(df):
    secao("2. ANÁLISE EXPLORATÓRIA - PACIENTES COM x SEM AVC", True)
    com, sem = df[df["stroke"] == 1], df[df["stroke"] == 0]
    grupo = df["stroke"].map({0: "Sem AVC", 1: "Com AVC"})
    print(f"Com AVC: {len(com)} ({br(100 * len(com) / len(df))}%) | Sem AVC: {len(sem)}")
    print("-> Grupos de tamanhos muito diferentes: comparamos médias, medianas e "
          "percentuais, não contagens.")

    secao("2.1 Idade, nível médio de glicose e IMC")
    estat = (df.groupby(grupo)[["age", "avg_glucose_level", "bmi"]]
             .agg(["mean", "median", "std", "min", "max"]).T[["Sem AVC", "Com AVC"]])
    estat.columns.name = None
    estat = estat.rename(index={"age": "Idade (anos)", "avg_glucose_level": "Glicose (mg/dL)",
                                "bmi": "IMC", "mean": "média", "median": "mediana",
                                "std": "desvio padrão", "min": "mínimo", "max": "máximo"})
    mostrar(estat)
    dif = com["age"].mean() - sem["age"].mean()
    print(f"\n* Idade: {br(com['age'].mean())} x {br(sem['age'].mean())} anos em média "
          f"({br(dif)} anos a mais) - a maior diferença entre todas as características.")
    print(f"* Glicose: média {br(com['avg_glucose_level'].mean())} x "
          f"{br(sem['avg_glucose_level'].mean())} mg/dL. A média do grupo com AVC é puxada "
          f"por valores altos: glicose >= 200 em {br((com['avg_glucose_level'] >= 200).mean() * 100)}%"
          f" dos pacientes com AVC x {br((sem['avg_glucose_level'] >= 200).mean() * 100)}% dos sem AVC.")
    originais = df[df["bmi_imputado"] == 0].groupby("stroke")["bmi"].mean()
    print(f"* IMC: {br(com['bmi'].mean())} x {br(sem['bmi'].mean())} - diferença pequena. "
          f"Só com os valores originais: {br(originais[1])} x {br(originais[0])}, "
          "então a imputação não mudou a conclusão.")

    secao("2.2 Hipertensão e doença cardíaca")
    binarias = {"hypertension": "Hipertensão", "heart_disease": "Doença cardíaca"}
    tabela = (df.groupby(grupo)[list(binarias)].mean().T * 100)[["Sem AVC", "Com AVC"]]
    tabela.columns = ["% entre SEM AVC", "% entre COM AVC"]
    tabela["taxa de AVC se tem (%)"] = [df.loc[df[c] == 1, "stroke"].mean() * 100 for c in binarias]
    tabela["taxa de AVC se não tem (%)"] = [df.loc[df[c] == 0, "stroke"].mean() * 100
                                            for c in binarias]
    tabela.index = list(binarias.values())
    mostrar(tabela)
    for nome, l in tabela.iterrows():
        print(f"* {nome}: presente em {br(l.iloc[1])}% dos pacientes com AVC x {br(l.iloc[0])}% "
              f"dos sem AVC. Quem tem: {br(l.iloc[2])}% de AVC, cerca de "
              f"{br(l.iloc[2] / l.iloc[3])} vezes a taxa de quem não tem.")

    secao("2.3 Situação de tabagismo")
    tabaco = df["smoking_status"].map(TABAGISMO).rename("tabagismo")
    resumo = df.groupby(tabaco).agg(pacientes=("stroke", "size"), casos_avc=("stroke", "sum"),
                                    taxa_avc_pct=("stroke", "mean"), idade_media=("age", "mean"))
    resumo["taxa_avc_pct"] *= 100
    distrib = pd.crosstab(tabaco, grupo, normalize="columns") * 100
    resumo["% do grupo SEM AVC"] = distrib["Sem AVC"]
    resumo["% do grupo COM AVC"] = distrib["Com AVC"]
    mostrar(resumo.sort_values("taxa_avc_pct", ascending=False).rename(columns=NOMES))
    menores = (df.loc[tabaco == "Desconhecido", "age"] < 18).mean() * 100
    print(f"* Ex-fumantes têm a maior taxa de AVC, mas também são o grupo mais velho "
          f"({br(resumo.loc['Ex-fumante', 'idade_media'])} anos): parte da diferença pode vir da idade.")
    print(f"* 'Desconhecido' tem a menor taxa porque concentra crianças: {br(menores)}% "
          "dos pacientes dessa categoria têm menos de 18 anos.")
    print("* Fumantes e quem nunca fumou têm taxas próximas: sem efeito isolado claro nesta base.")

    secao("2.4 Taxa de AVC por faixa etária")
    faixas = df.groupby("faixa_etaria", observed=True)["stroke"].agg(
        pacientes="size", casos_avc="sum", taxa_avc_pct="mean")
    faixas["taxa_avc_pct"] *= 100
    mostrar(faixas.rename(columns=NOMES))
    print(f"* A taxa sobe com a idade: de {br(faixas['taxa_avc_pct'].iloc[0])}% (0-19) "
          f"para {br(faixas['taxa_avc_pct'].iloc[-1])}% (80+).")

    secao("2.5 Taxa de AVC (%) por faixa etária, separando hipertensos")
    cruzada = df.pivot_table(index="faixa_etaria", columns="hypertension", values="stroke",
                             aggfunc="mean", observed=True) * 100
    cruzada.columns = ["Sem hipertensão", "Com hipertensão"]
    mostrar(cruzada)
    print("* Hipertensos são em geral mais velhos, então comparamos dentro da mesma faixa: "
          "a partir dos 50 anos eles têm taxa maior em todas as faixas.")

    secao("Resumo")
    print("1. Idade é a característica com maior diferença entre os grupos.\n"
          "2. Hipertensão e doença cardíaca são bem mais frequentes no grupo com AVC.\n"
          "3. Glicose média é maior no grupo com AVC, puxada por valores >= 200.\n"
          "4. IMC quase não difere entre os grupos.\n"
          "5. Tabagismo não mostra efeito claro quando se considera a idade.\n\n"
          "OBSERVAÇÃO: são associações observadas NESTE dataset. Diferença entre os grupos "
          "não prova causa; outras variáveis (como a idade) podem explicar parte delas.")
    return faixas


def salvar(fig, ax, titulo, eixo_x, eixo_y, arquivo, legenda="upper left"):
    ax.set_title(titulo, fontsize=14, fontweight="bold")
    ax.set_xlabel(eixo_x, fontsize=12)
    ax.set_ylabel(eixo_y, fontsize=12)
    ax.tick_params(labelsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=11, loc=legenda)
    fig.tight_layout()
    fig.savefig(PASTA / arquivo, dpi=150)
    print("Gráfico salvo:", arquivo)
    if MOSTRAR_GRAFICOS:
        plt.show()
    plt.close(fig)


def gerar_graficos(df, faixas):
    secao("3. VISUALIZAÇÃO DOS DADOS", True)

    idades = df.loc[df["stroke"] == 1, "age"]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(idades, bins=range(0, 90, 5), color=COR_COM, edgecolor="white")
    ax.axvline(idades.mean(), color="black", ls="--", label=f"Média: {br(idades.mean())} anos")
    ax.axvline(idades.median(), color="black", ls=":", lw=2,
               label=f"Mediana: {br(idades.median())} anos")
    ax.set_xticks(range(0, 90, 5))
    salvar(fig, ax, f"Gráfico 1 – Distribuição das idades dos pacientes com AVC (n = {len(idades)})",
           "Idade (anos), em intervalos de 5 anos", "Número de pacientes com AVC",
           "grafico1_idades_avc.png")

    faixa_glicose = pd.cut(df["avg_glucose_level"], [0, 100, 140, 200, 1000], right=False,
                           labels=["< 100", "100 a < 140", "140 a < 200", "≥ 200"])
    percentual = pd.crosstab(faixa_glicose, df["stroke"], normalize="columns") * 100
    posicoes = list(range(len(percentual)))
    fig, ax = plt.subplots(figsize=(10, 5))
    for deslocamento, s, cor, nome in [(-0.2, 0, COR_SEM, "Sem AVC"), (0.2, 1, COR_COM, "Com AVC")]:
        barras = ax.bar([x + deslocamento for x in posicoes], percentual[s], width=0.4,
                        color=cor, label=f"{nome} (n = {br((df['stroke'] == s).sum(), 0)})")
        ax.bar_label(barras, [f"{br(v)}%" for v in percentual[s]], padding=3, fontsize=10)
    ax.set_xticks(posicoes)
    ax.set_xticklabels(percentual.index.astype(str))
    ax.set_ylim(0, percentual.values.max() * 1.15)
    salvar(fig, ax, "Gráfico 2 – Nível médio de glicose: pacientes com e sem AVC",
           "Nível médio de glicose (mg/dL)", "Pacientes do grupo (%)",
           "grafico2_glicose_avc.png", legenda="upper right")

    taxa_geral = df["stroke"].mean() * 100
    fig, ax = plt.subplots(figsize=(10, 5))
    barras = ax.bar(faixas.index.astype(str), faixas["taxa_avc_pct"], color=COR_COM,
                    alpha=0.85, edgecolor="black")
    ax.axhline(taxa_geral, color="gray", ls="--", label=f"Taxa geral da base: {br(taxa_geral)}%")
    rotulos = [f"{br(t)}%\n{c} de {br(p, 0)}" for t, c, p in
               zip(faixas["taxa_avc_pct"], faixas["casos_avc"], faixas["pacientes"])]
    ax.bar_label(barras, rotulos, padding=3, fontsize=10)
    ax.set_ylim(0, faixas["taxa_avc_pct"].max() * 1.25)
    salvar(fig, ax, "Gráfico 3 – Percentual de pacientes com AVC por faixa etária",
           "Faixa etária (anos)", "Pacientes com AVC na faixa (%)", "grafico3_taxa_avc_faixa.png")
    print("\nJustificativa do Gráfico 3: o Gráfico 1 conta casos, mas não considera quantos "
          "pacientes há em cada faixa. O Gráfico 3 mostra a PROPORÇÃO de AVC dentro de cada "
          "faixa, revelando como a ocorrência cresce com a idade.")


COLUNAS = ["id", "gender", "age", "hypertension", "heart_disease", "ever_married",
           "work_type", "residence_type", "avg_glucose_level", "bmi", "bmi_imputado",
           "smoking_status", "faixa_etaria", "stroke"]

CONSULTAS = {
    "1 - Quantidade total de registros":
        "SELECT COUNT(*) AS total_registros FROM pacientes",
    "2a - Quantidade de pacientes com AVC":
        "SELECT COUNT(*) AS pacientes_com_avc FROM pacientes WHERE stroke = 1",
    "2b - Pacientes com AVC (10 mais velhos)":
        """SELECT id, gender, age, hypertension, heart_disease, avg_glucose_level, bmi,
                  smoking_status
           FROM pacientes WHERE stroke = 1 ORDER BY age DESC LIMIT 10""",
    "3 (equipe) - Taxa de AVC com e sem hipertensão":
        """SELECT CASE hypertension WHEN 1 THEN 'Com hipertensão'
                                    ELSE 'Sem hipertensão' END AS grupo,
                  COUNT(*) AS pacientes, SUM(stroke) AS casos_avc,
                  ROUND(100.0 * SUM(stroke) / COUNT(*), 2) AS taxa_avc_pct,
                  ROUND(AVG(age), 1) AS idade_media
           FROM pacientes GROUP BY hypertension ORDER BY hypertension DESC""",
    "4 (extra) - Taxa de AVC por tabagismo":
        """SELECT smoking_status, COUNT(*) AS pacientes, SUM(stroke) AS casos_avc,
                  ROUND(100.0 * SUM(stroke) / COUNT(*), 2) AS taxa_avc_pct,
                  ROUND(AVG(age), 1) AS idade_media
           FROM pacientes GROUP BY smoking_status ORDER BY taxa_avc_pct DESC""",
    "5 (extra) - AVC com hipertensão e glicose >= 200":
        """SELECT COUNT(*) AS pacientes, ROUND(AVG(age), 1) AS idade_media,
                  ROUND(AVG(avg_glucose_level), 1) AS glicose_media
           FROM pacientes
           WHERE stroke = 1 AND hypertension = 1 AND avg_glucose_level >= 200""",
}


def banco_de_dados(df):
    secao("4. PERSISTÊNCIA DOS DADOS - SQLITE", True)
    con = sqlite3.connect(DB)
    con.execute("DROP TABLE IF EXISTS pacientes")
    con.execute("""
        CREATE TABLE pacientes (
            id                INTEGER PRIMARY KEY,
            gender            TEXT    NOT NULL,
            age               REAL    NOT NULL,
            hypertension      INTEGER NOT NULL CHECK (hypertension IN (0, 1)),
            heart_disease     INTEGER NOT NULL CHECK (heart_disease IN (0, 1)),
            ever_married      TEXT,
            work_type         TEXT,
            residence_type    TEXT,
            avg_glucose_level REAL    NOT NULL,
            bmi               REAL    NOT NULL,
            bmi_imputado      INTEGER NOT NULL CHECK (bmi_imputado IN (0, 1)),
            smoking_status    TEXT    NOT NULL,
            faixa_etaria      TEXT    NOT NULL,
            stroke            INTEGER NOT NULL CHECK (stroke IN (0, 1))
        )""")
    df[COLUNAS].astype({"faixa_etaria": str}).to_sql("pacientes", con, if_exists="append",
                                                     index=False)
    con.commit()
    print(f"Banco {DB.name} criado | tabela pacientes | {len(df)} registros inseridos")

    for titulo, sql in CONSULTAS.items():
        print(f"\n>> Consulta {titulo}\nSQL: {' '.join(sql.split())}")
        mostrar(pd.read_sql_query(sql, con), casas=None, index=False)
    con.close()


if __name__ == "__main__":
    if not CSV.exists():
        raise SystemExit(f"ERRO: {CSV.name} não encontrado em {PASTA}")
    dados = carregar_e_tratar()
    faixas_etarias = analisar(dados)
    gerar_graficos(dados, faixas_etarias)
    banco_de_dados(dados)
    print("\nProcessamento concluído com sucesso.")