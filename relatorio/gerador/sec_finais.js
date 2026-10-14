// Seção 5 – avaliação final, custo × efetividade e análise de erros
const fs = require("fs");
const { P, H1, H2, H3, bullets, figure, table, nextFig, nextTab } = require("./lib");
const readCsv = require("./csv");
const FIG = (f) => `${require("path").join(__dirname, "..", "..", "outputs")}/figures/${f}.png`;
const TAB = (f) => `${require("path").join(__dirname, "..", "..", "outputs")}/tables/${f}`;
const n = (x, k = 2) => Number(x).toLocaleString("pt-BR", { minimumFractionDigits: k, maximumFractionDigits: k });
const int = (x) => Math.round(Number(x)).toLocaleString("pt-BR");
const sg = (x, k = 1) => (Number(x) > 0 ? "+" : "") + n(x, k);
const NOME = { media_treino: "Média do treino", linear: "Regressão linear", A_melhor: "A – melhor (2×658)", B_compacta: "B – compacta (1×173)", C_contraste: "C – profunda (380→47)" };

module.exports = function finais() {
  const agg = readCsv(TAB("resultados_finais.csv"));
  const g = (m, s) => agg.find((r) => r.modelo === m && r.split === s);
  const E = JSON.parse(fs.readFileSync(TAB("analise_erros.json"), "utf8"));
  const dec = readCsv(TAB("erro_decis.csv"));
  const grp = readCsv(TAB("erro_por_grupo.csv"));
  const uf = readCsv(TAB("erro_uf.csv"));
  const renda = readCsv(TAB("erro_renda.csv"));
  const out = [];
  const A22 = g("A_melhor", "test22"), A23 = g("A_melhor", "test23"), L22 = g("linear", "test22"), L23 = g("linear", "test23");
  const B22 = g("B_compacta", "test22"), B23 = g("B_compacta", "test23"), C22 = g("C_contraste", "test22"), C23 = g("C_contraste", "test23");
  const M22 = g("media_treino", "test22"), M23 = g("media_treino", "test23"), Av = g("A_melhor", "val");

  out.push(H1("5. Avaliação final no conjunto de teste e análise de custo × efetividade"));
  out.push(H2("5.1 Desempenho no teste"));
  out.push(P(`As três arquiteturas e a regressão linear foram retreinadas no treino completo (1,87 milhão), com early stopping na validação (paciência 5, até 40 épocas), três sementes cada, e avaliadas **uma única vez** nos dois conjuntos de teste. A Tabela ${nextTab()} traz a média das três sementes.`));
  const rows = ["media_treino", "linear", "A_melhor", "B_compacta", "C_contraste"].map((m) => {
    const v = g(m, "val"), t2 = g(m, "test22"), t3 = g(m, "test23");
    return [NOME[m], n(v.rmse), n(t2.rmse), n(t2.mae), n(t2.r2, 3), n(t3.rmse), n(t3.mae), n(t3.r2, 3), sg(t3.vies)];
  });
  out.push(...table("Desempenho final (média de 3 sementes). RMSE, MAE e viés em pontos; viés = previsto − real.",
    ["Modelo", "RMSE val.", "RMSE teste-22", "MAE teste-22", "R² teste-22", "RMSE teste-23", "MAE teste-23", "R² teste-23", "Viés teste-23"],
    rows, [0.20, 0.09, 0.10, 0.10, 0.09, 0.10, 0.10, 0.10, 0.12], ["l", "r", "r", "r", "r", "r", "r", "r", "r"], { size: 15,
      note: "Desvio-padrão do RMSE entre sementes ≤ 0,04 ponto em todos os modelos e partições." }));
  out.push(...bullets([
    `**A MLP supera as referências.** No teste-2022, o modelo A erra ${n(A22.rmse)} pontos de RMSE (MAE ${n(A22.mae)}; R² = ${n(A22.r2, 3)}), contra ${n(M22.rmse)} de prever sempre a média e ${n(L22.rmse)} da regressão linear com as mesmas entradas. A não linearidade reduz o RMSE em ${n(L22.rmse - A22.rmse)} ponto e aumenta o R² em ${n(A22.r2 - L22.r2, 3)}. É um ganho real e estável entre sementes, mas modesto: a maior parte do que dá para extrair dessas entradas uma regressão linear já captura.`,
    `**Validação, teste-2022 e treino concordam.** RMSE de ${n(Av.rmse)} na validação e ${n(A22.rmse)} no teste, e as curvas de treino terminam ~0,4 ponto abaixo da validação (Figura ${nextFig()}). Não há overfitting relevante nem sinal de que a seleção de modelos se ajustou à validação.`,
    `**Retreinar com todos os dados valeu.** O modelo A caiu de ${n(68.84)} (Optuna, 400 mil) para ${n(Av.rmse)} na validação (−0,44 ponto), mais do que separa as três arquiteturas entre si. Isso repete a lição de E1: aqui, quantidade de dados pesa mais que arquitetura.`,
    `**No ano seguinte, o modelo mantém o poder explicativo.** No teste-2023 o RMSE sobe para ${n(A23.rmse)}, mas o R² quase não muda (${n(A22.r2, 3)} → ${n(A23.r2, 3)}). O aumento vem da maior dispersão das notas de 2023 (desvio 93,0 contra 87,8; Seção 2.4), não de uma relação perfil → nota diferente. O viés é pequeno (${n(A23.vies)} ponto): em 2023 o modelo subestima levemente as notas, e zerá-lo mudaria o RMSE em menos de 0,05.`,
  ]));
  out.push(...figure(FIG("final_curvas"), "Curvas de treino e validação no treino completo (semente 0). O ponto marca a época restaurada pelo early stopping."));
  out.push(P(`Nos modelos A e B, a melhor época (37–38 em média) ficou perto do limite de 40, com a taxa de aprendizado já reduzida várias vezes pelo scheduler e a validação praticamente estável. Mais épocas trariam, no máximo, ganhos de centésimos. O modelo C, mais profundo, parou na época 23 e é o único com separação visível entre as curvas (≈1 ponto no fim).`));

  out.push(H2("5.2 Custo × efetividade"));
  out.push(...table("Custo computacional dos modelos finais (média de 3 sementes; GPU Apple M2).",
    ["Modelo", "Parâmetros", "Treino (s)", "Inferência (s / 100 mil)", "RMSE teste-23", "Δ RMSE vs. A"],
    ["linear", "A_melhor", "B_compacta", "C_contraste"].map((m) => { const t = g(m, "test23");
      return [NOME[m], int(t.n_params), int(t.treino_s), n(t.inferencia_100k_s, 3), n(t.rmse), m === "A_melhor" ? "–" : sg(t.rmse - A23.rmse, 2)]; }),
    [0.26, 0.14, 0.12, 0.18, 0.15, 0.15], ["l", "r", "r", "r", "r", "r"]));
  out.push(P(`A Figura ${nextFig()} resume o compromisso. O modelo **B (compacto)** tem ${n(A23.n_params / B23.n_params, 1)} vezes menos parâmetros que A, treina ${n(A23.treino_s / B23.treino_s, 1)} vezes mais rápido (${int(B23.treino_s)} s contra ${int(A23.treino_s)} s) e infere ${n(A23.inferencia_100k_s / B23.inferencia_100k_s, 0)} vezes mais rápido, ao custo de ${n(B23.rmse - A23.rmse, 2)} ponto de RMSE em 2023 (${n(B22.rmse - A22.rmse, 2)} em 2022). A diferença é estável entre sementes (desvio ≤ 0,04), mas pequena demais para ter consequência prática, já que é menos de 0,2% do erro. O modelo **C (profundo)** custa 2,4 vezes o tempo de B e empata com ele: **profundidade não comprou nada**, de novo em linha com E1 e com a importância quase nula de \"camadas\" no Optuna. A regressão linear é ~14 vezes mais barata que B e ${n(L23.rmse - B23.rmse, 1)} pontos pior. **Para uso real, B é a melhor escolha**; A só se justifica se cada décimo de ponto importar.`));
  out.push(...figure(FIG("final_custo_efetividade"), "RMSE no teste-2023 por número de parâmetros e por tempo de treino (barras = desvio entre sementes, menores que o marcador)."));

  out.push(H2("5.3 Análise de erros: onde e como o modelo falha"));
  out.push(P(`Analisamos o modelo A (melhor na validação) nos 2,66 milhões de participantes de 2023. ${n(100 * E.frac_erro_abs_lt_50, 0)}% das predições erram menos de 50 pontos, e ${n(100 * E.frac_erro_abs_gt_100, 0)}% erram mais de 100.`));
  out.push(H3("Regressão à média"));
  const d1 = dec[0], d10 = dec[dec.length - 1];
  out.push(P(`O principal padrão de erro é a **compressão das predições** (Figuras ${nextFig()} e ${nextFig() + 1}). As notas reais vão de ${int(246)} a ${int(863)} (desvio ${n(E.std_real, 1)}), e as previstas só de ${int(E.pred_min)} a ${int(E.pred_max)} (desvio ${n(E.std_pred, 1)}). No decil mais baixo das notas reais (média ${n(d1.nota_real, 0)}), o modelo prevê em média ${n(d1.prevista, 0)}, superestimando em ${n(d1.vies, 0)} pontos. No decil mais alto (média ${n(d10.nota_real, 0)}), prevê ${n(d10.prevista, 0)} e subestima em ${n(-d10.vies, 0)}. Isso **não é um defeito de treino**: é o comportamento ótimo de um estimador de erro quadrático quando as entradas explicam só ~39% da variância. A melhor predição é a média condicional ao perfil, e quem vai muito bem ou muito mal se afasta dessa média por motivos que as entradas não registram.`));
  out.push(...figure(FIG("erro_hexbin"), "Nota média prevista × real no teste-2023 (escala log de densidade). A linha é a predição perfeita.", 0.62));
  out.push(...figure(FIG("erro_decis"), "Nota média real e prevista por decil da nota real (teste-2023).", 0.75));
  out.push(H3("Erro por grupo"));
  const G = (grupo, cat) => grp.find((r) => r.grupo === grupo && r.categoria === cat);
  const rz = G("Redação", "nota 0"), rok = G("Redação", "nota > 0");
  const ind = G("Cor/raça (não usada no modelo)", "indígena"), pre = G("Cor/raça (não usada no modelo)", "preta"), bra = G("Cor/raça (não usada no modelo)", "branca"), par = G("Cor/raça (não usada no modelo)", "parda");
  const fem = G("Sexo (não usado no modelo)", "feminino"), mas = G("Sexo (não usado no modelo)", "masculino");
  const priv = G("Tipo de escola", "privada"), pub = G("Tipo de escola", "pública");
  const grows = grp.map((r) => [r.grupo, r.categoria, int(r.n), n(r.real, 1), sg(r.vies), n(r.rmse, 1)]);
  out.push(...table("Erro por grupo no teste-2023 (modelo A). Viés = previsto − real (positivo = superestima).",
    ["Grupo", "Categoria", "Participantes", "Nota real média", "Viés", "RMSE"],
    grows, [0.30, 0.16, 0.14, 0.14, 0.12, 0.14], ["l", "l", "r", "r", "r", "r"], { size: 15 }));
  out.push(...bullets([
    `**Redação zero é o maior modo de falha.** Os ${int(rz.n)} participantes com redação 0 (3,4%) têm nota média real de ${n(rz.real, 0)}, e o modelo os superestima em ${n(rz.vies, 0)} pontos (RMSE ${n(rz.rmse, 0)}, contra ${n(rok.rmse, 1)} dos demais). Zerar a redação (em branco, fuga ao tema, cópia) é um evento do dia da prova, e o perfil socioeconômico quase não o antecipa.`,
    `**Cor/raça: viés residual, mesmo sem usar a variável.** O modelo superestima indígenas (${sg(ind.vies)}) e pretos (${sg(pre.vies)}) e subestima brancos (${sg(bra.vies)}); pardos ficam perto de zero (${sg(par.vies)}). Renda, escolaridade dos pais e município captam a maior parte da diferença entre grupos, mas não toda. Na prática, se o modelo fosse usado para **direcionar apoio**, ele daria prioridade um pouco menor justamente a indígenas e pretos, cujas notas reais são mais baixas que as previstas. Isso deve ser corrigido (por exemplo, com calibração por grupo) antes de qualquer uso.`,
    `**Sexo:** o viés é pequeno (${sg(fem.vies)} feminino, ${sg(mas.vies)} masculino), mas o RMSE é maior para homens (${n(mas.rmse, 1)} contra ${n(fem.rmse, 1)}), porque as notas masculinas são mais dispersas.`,
    `**Escola privada:** subestimada em ${n(-priv.vies, 1)} pontos (pública: ${sg(pub.vies)}), sinal de que \"privada\" agrupa escolas de qualidade muito diferente e de que falta o efeito da escola em si (sem código de escola em 2023).`,
    `**UF e renda:** o viés por UF fica entre ${sg(uf[0].vies)} (${uf[0].SG_UF_PROVA}) e ${sg(uf[uf.length - 1].vies)} (${uf[uf.length - 1].SG_UF_PROVA}), e o RMSE entre ${n(Math.min(...uf.map(r => +r.rmse)), 1)} e ${n(Math.max(...uf.map(r => +r.rmse)), 1)} (o maior no CE). Por faixa de renda, o RMSE é quase constante (${n(Math.min(...renda.map(r => +r.rmse)), 1)}–${n(Math.max(...renda.map(r => +r.rmse)), 1)}), e o viés vai de ${sg(renda[0].vies)} na faixa sem renda a cerca de −6 nas faixas altas (Figuras ${nextFig()} e ${nextFig() + 1}). Parte do leve viés negativo geral de 2023 vem daí.`,
  ]));
  out.push(...figure(FIG("erro_uf"), "Viés médio por UF da prova no teste-2023 (azul = superestima; laranja = subestima)."));
  out.push(...figure(FIG("erro_renda"), "RMSE e viés por faixa de renda familiar no teste-2023.", 0.8));
  return out;
};
