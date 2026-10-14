// Seções 6 (limitações), 7 (conclusões), referências e apêndices
const fs = require("fs");
const { P, H1, H2, bullets, numbered, table } = require("./lib");
const readCsv = require("./csv");
const limitacoes = require("./sec_limitacoes");
const TAB = (f) => `${require("path").join(__dirname, "..", "..", "outputs")}/tables/${f}`;
const n = (x, k = 2) => Number(x).toLocaleString("pt-BR", { minimumFractionDigits: k, maximumFractionDigits: k });
const int = (x) => Math.round(Number(x)).toLocaleString("pt-BR");
const sg = (x, k = 1) => (Number(x) > 0 ? "+" : "") + n(x, k);
const sci = (x) => { const v = Number(x); const e = Math.floor(Math.log10(v)); const m = v / 10 ** e;
  const sup = String(e).replace("-", "⁻").replace(/\d/g, (d) => "⁰¹²³⁴⁵⁶⁷⁸⁹"[d]); return `${n(m, 1)}·10${sup}`; };

module.exports = function conclusao() {
  const E = JSON.parse(fs.readFileSync(TAB("analise_erros.json"), "utf8"));
  const dec = readCsv(TAB("erro_decis.csv"));
  const grp = readCsv(TAB("erro_por_grupo.csv"));
  const G = (grupo, cat) => grp.find((r) => r.grupo === grupo && r.categoria === cat);
  const cr = "Cor/raça (não usada no modelo)";
  const F = {
    r2: "39%", std_pred: n(E.std_pred, 1), std_real: n(E.std_real, 1),
    vies_d1: sg(dec[0].vies, 0), vies_d10: sg(dec[dec.length - 1].vies, 0),
    rmse_red0: n(G("Redação", "nota 0").rmse, 0), rmse_red_ok: n(G("Redação", "nota > 0").rmse, 1),
    vies_ind: sg(G(cr, "indígena").vies), vies_pre: sg(G(cr, "preta").vies), vies_bra: sg(G(cr, "branca").vies),
  };
  const out = [...limitacoes(F)];

  out.push(H1("7. Conclusões"));
  out.push(P("Construímos e avaliamos MLPs para prever a nota média no ENEM só com informações anteriores à prova, usando os microdados de 2022 (treino) e 2023 (teste no ano seguinte). As principais conclusões:"));
  out.push(...numbered([
    "**Dá para antecipar cerca de 39% da variação da nota, e não muito mais.** A melhor MLP chega a RMSE de 68,4 pontos (R² = 0,392) no teste-2022 e mantém o mesmo poder explicativo em 2023 (R² = 0,385), contra 87,8 de prever a média e 70,2 da regressão linear. O limite vem dos dados: pessoas com o mesmo perfil variam ~73 pontos entre si.",
    "**A não linearidade ajuda, mas a capacidade extra não.** Uma única camada oculta já captura o ganho sobre o modelo linear (~1,7 ponto). Profundidade e largura são as variáveis menos importantes da busca (fANOVA ≤ 0,03), e a rede profunda de 4 camadas empatou com a de 1 camada.",
    "**Regularização é o que separa um bom treino de um ruim, e a estratégia certa depende do volume de dados.** Sem regularização, redes largas decoram o treino (RMSE de treino 35 contra 85 na validação). Com 500 mil exemplos, L2/L1 fortes ou dropout eliminam o overfitting. Com 20 mil, só o early stopping funciona, e as penalidades chegaram a piorar o resultado. No Optuna, dropout (~0,4) com LayerNorm e L2 fraco formou a região ótima; L2 forte levou a subajuste.",
    "**Mais dados valem mais que mais arquitetura.** Passar de 20 mil para 500 mil exemplos reduziu o RMSE em 1,6 ponto, e retreinar com 1,87 milhão reduziu mais 0,4. Nenhuma mudança de arquitetura chegou perto disso.",
    "**Custo × efetividade favorece a rede compacta.** A MLP de 1 camada com 173 neurônios tem 9,3 vezes menos parâmetros e treina 3,8 vezes mais rápido que a melhor, perdendo só 0,1 ponto de RMSE.",
    "**O modelo serve para grupos, não para indivíduos.** As predições são comprimidas para o centro (regressão à média), falham muito em quem zera a redação e têm viés residual por cor/raça. São úteis para apontar perfis, escolas e municípios que precisam de apoio, desde que calibradas por grupo, e inadequadas para decisões sobre uma pessoa.",
  ]));
  out.push(P("**Trabalhos futuros.** (i) Incorporar características da escola, juntando os microdados de 2024 com o Censo Escolar pelo código da escola, numa formulação por escola, já que as notas de 2024 não se ligam ao questionário; (ii) prever as cinco notas de uma vez, com uma saída multitarefa; (iii) modelar a redação zero separadamente (classificação + regressão); (iv) calibração por grupo para reduzir o viés por cor/raça; (v) comparar com modelos de árvores (gradient boosting), referência forte em dados tabulares."));

  out.push(H1("Referências"));
  out.push(...[
    "INSTITUTO NACIONAL DE ESTUDOS E PESQUISAS EDUCACIONAIS ANÍSIO TEIXEIRA (INEP). **Microdados do ENEM 2022** e **Microdados do ENEM 2023**. Brasília: Inep. Disponível em: https://dados.gov.br/dados/conjuntos-dados/inep-microdados-do-enem.",
    "INEP. **Microdados ENEM 2024 – Leia-me**. Brasília: Inep, jun. 2025 (mudança de formato: arquivos de participantes e resultados sem chave de ligação).",
    "AKIBA, T. et al. Optuna: a next-generation hyperparameter optimization framework. In: **Proc. 25th ACM SIGKDD**, 2019.",
    "BERGSTRA, J. et al. Algorithms for hyper-parameter optimization (TPE). In: **Advances in Neural Information Processing Systems 24**, 2011.",
    "LOSHCHILOV, I.; HUTTER, F. Decoupled weight decay regularization. In: **ICLR**, 2019.",
    "SRIVASTAVA, N. et al. Dropout: a simple way to prevent neural networks from overfitting. **Journal of Machine Learning Research**, v. 15, 2014.",
    "IOFFE, S.; SZEGEDY, C. Batch normalization. In: **ICML**, 2015. · BA, J. L.; KIROS, J. R.; HINTON, G. E. Layer normalization. arXiv:1607.06450, 2016.",
    "GUO, C.; BERKHAHN, F. Entity embeddings of categorical variables. arXiv:1604.06737, 2016.",
    "HUTTER, F.; HOOS, H.; LEYTON-BROWN, K. An efficient approach for assessing hyperparameter importance (fANOVA). In: **ICML**, 2014.",
    "PASZKE, A. et al. PyTorch: an imperative style, high-performance deep learning library. In: **NeurIPS**, 2019.",
  ].map((r) => P(r, { align: undefined, spacing: { after: 80, line: 276 } })));

  out.push(H1("Apêndice A – Registro da busca de hiperparâmetros"));
  out.push(P("Os 15 melhores trials completos do estudo `enem_mlp` (60 trials; TPE multivariado com 12 trials iniciais aleatórios, semente 42; MedianPruner com 8 trials de aquecimento e 4 épocas de warmup; 400 mil exemplos de treino e 100 mil de validação). O registro completo, com valores por época e trials podados, está em `outputs/optuna.db` e `outputs/tables/optuna_trials.csv`; o histórico de cada trial, em `outputs/runs/optuna/`."));
  const tr = readCsv(TAB("optuna_trials.csv")).filter((t) => t.state === "COMPLETE").sort((a, b) => a.value - b.value).slice(0, 15);
  out.push(...table("Os 15 melhores trials do Optuna (RMSE de validação em pontos).",
    ["#", "RMSE val.", "RMSE treino", "Camadas", "Ativ.", "Norm.", "Dropout", "L2", "LR", "Lote", "Emb.", "Parâm."],
    tr.map((t) => [t.number, n(t.value), n(t.user_attrs_train_rmse), t.user_attrs_hidden.replace(/[\[\]]/g, "").replace(/, /g, "→"), t.params_ativacao,
      t.params_normalizacao === "layer" ? "Layer" : t.params_normalizacao, n(t.params_dropout), sci(t.params_l2), sci(t.params_lr), t.params_batch_size, t.params_emb_municipio, int(t.user_attrs_n_params)]),
    [0.04, 0.08, 0.08, 0.16, 0.06, 0.07, 0.08, 0.09, 0.09, 0.07, 0.06, 0.12], ["r", "r", "r", "l", "l", "l", "r", "r", "r", "r", "r", "r"], { size: 14 }));

  out.push(H1("Apêndice B – Reprodutibilidade"));
  out.push(P("Todo o código está no repositório (estrutura e comandos no `README.md`). A sequência completa:"));
  out.push(...bullets([
    "`scripts/01_preparar_dados.py`: lê os zips do INEP, filtra os registros e grava os parquets (~1 min).",
    "`pipeline.build_cache()`: divide os dados, ajusta o pré-processamento no treino e grava os dados codificados.",
    "`scripts/02_capacidade.py`, `03_regularizacao.py`, `03b_diagnostico_l2.py`: experimentos E1 e E2 (~1 h no M2).",
    "`scripts/04_optuna.py 60`: experimento E3 (~25 min); `scripts/05_modelos_finais.py`: E4 (~80 min).",
    "`notebooks/01_analise_exploratoria.ipynb` e `02_experimentos_e_resultados.ipynb`: geram todas as tabelas e figuras deste relatório a partir dos registros.",
    "Sementes fixas (42 para divisão e amostras; 0–2 nos modelos finais). Na GPU do Mac (MPS), algumas operações não são determinísticas; diferenças entre execuções ficam abaixo de 0,05 ponto no melhor ponto da validação.",
  ]));
  return out;
};
