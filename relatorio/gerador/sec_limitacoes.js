// Seção 6 – limitações (os números de erro por grupo entram via parâmetro)
const { P, H1, H2, bullets } = require("./lib");

module.exports = function limitacoes(F) {
  // F: fatos calculados em sec_finais (strings já formatadas)
  const out = [];
  out.push(H1("6. Limitações, incertezas e situações de falha"));
  out.push(...bullets([
    `**Erro irredutível alto.** As entradas explicam ~${F.r2} da variância. Isso não é falha de ajuste: pessoas com o mesmo perfil de entrada diferem em ~73 pontos (Seção 2.3), e E1–E4 mostram um platô: nenhuma arquitetura, regularização ou combinação de hiperparâmetros ficou muito abaixo de ~68,4 de RMSE. O que falta são informações que não estão nos microdados: qualidade da escola, histórico de notas, preparação, motivação e as condições no dia da prova.`,
    `**Regressão à média (principal modo de falha).** As predições variam bem menos que as notas reais (desvio de ${F.std_pred} contra ${F.std_real} pontos em 2023). O modelo **superestima quem vai mal e subestima quem vai bem**: em 2023, erra em média ${F.vies_d1} pontos no decil mais baixo e ${F.vies_d10} no mais alto. Por isso o modelo serve para **médias de grupos** (escolas, municípios, perfis), não para prever a nota de uma pessoa. Usá-lo para decisões individuais, como seleção ou alocação de bolsas, seria inadequado.`,
    `**Redação zero e eventos do dia da prova.** Quem zera a redação (em branco, fuga ao tema, cópia; 3–4% dos participantes) perde ~125 pontos na média, e o perfil socioeconômico quase não antecipa isso. É o grupo de maior erro (RMSE ${F.rmse_red0} em 2023, contra ${F.rmse_red_ok} dos demais).`,
    `**Viés de seleção.** O modelo só vale para quem fez as quatro provas (~67% dos inscritos). Os faltantes têm, provavelmente, perfil diferente, e o modelo não diz nada sobre a nota que teriam.`,
    "**Qualidade e granularidade das entradas.** O questionário é autodeclarado; renda e idade vêm em faixas; o município é o da prova (o de residência foi retirado pelo INEP); os campos da escola só existem para os concluintes do ano, e o código da escola não existe em 2022–2023. Efeitos de escola, que a literatura aponta como grandes, ficam fora do modelo.",
    `**Validade temporal limitada.** O teste em 2023 mostrou estabilidade da relação perfil → nota (R² praticamente igual), mas foi só um ano à frente. A partir de 2024, o INEP mudou o formato dos microdados e o questionário (23 perguntas em vez de 25), o que exige readaptar as entradas, e, sem notas ligadas ao questionário, não é mais possível avaliar o modelo em edições novas.`,
    `**Equidade.** Sexo e cor/raça não são entradas, mas o erro não é neutro em relação a eles: em 2023 o modelo superestima indígenas (${F.vies_ind}) e pretos (${F.vies_pre}) e subestima brancos (${F.vies_bra}) (Seção 5.3). Usado sem correção para priorizar apoio, ele subatenderia justamente os grupos com notas mais baixas que as previstas. Qualquer uso precisa de auditoria e calibração por grupo.`,
    `**Incerteza das comparações.** Com 100 mil exemplos de validação, o erro-padrão de um RMSE é de ~0,15 ponto. Diferenças menores que ~0,3 ponto entre configurações de E1–E3 não devem ser lidas como relevantes; parte dos 10 melhores trials do Optuna é empate estatístico. Os modelos finais usam 234 mil (teste-2022) e 2,66 milhões (teste-2023) exemplos e 3 sementes. A variação entre sementes (Tabela de resultados) é a melhor estimativa da instabilidade do treino. Além disso, a GPU do Mac (MPS) não é totalmente determinística: execuções repetidas da mesma configuração diferem nas épocas finais de redes que já decoraram o treino, mas não no melhor ponto.`,
    "**Custo da busca em amostra.** O Optuna usou 400 mil exemplos por custo. Com o treino completo, a região ótima pode mudar um pouco, e os modelos finais retreinados mostram se a ordem entre as arquiteturas se manteve.",
  ]));
  return out;
};
