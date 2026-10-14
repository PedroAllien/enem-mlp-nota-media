// Resumo executivo (início do relatório)
const { P, H1 } = require("./lib");
module.exports = function resumo() {
  return [
    H1("Resumo"),
    P("Este trabalho investiga quanto da nota média de um participante no ENEM pode ser prevista **antes da prova**, só com o questionário socioeconômico da inscrição, a situação escolar, a faixa etária e o local de prova. Usamos os microdados do INEP: 1,87 milhão de participantes de 2022 para treino, 234 mil para validação e 234 mil para teste, e todos os 2,66 milhões de participantes de 2023 como **teste temporal**. O modelo é uma MLP com entradas one-hot e um embedding para o município, construída em PyTorch."),
    P("Quatro experimentos estudaram (E1) profundidade e largura, (E2) sete estratégias de regularização em duas escalas de dados, (E3) uma busca de hiperparâmetros com Optuna (60 trials, TPE com poda) e (E4) a avaliação final de três arquiteturas selecionadas, com três sementes cada. A melhor MLP atinge **RMSE de 68,4 pontos e R² de 0,39** no teste de 2022, contra 87,8 de prever a média e 70,2 da regressão linear, e mantém o R² no ano seguinte (0,385). Uma rede de uma camada com 9 vezes menos parâmetros fica a 0,1 ponto disso. Regularização e volume de dados importam muito mais que profundidade e largura. O modelo comprime as predições para o centro (regressão à média) e tem viés residual por cor/raça, então serve para analisar grupos, não para decisões sobre indivíduos."),
  ];
};
