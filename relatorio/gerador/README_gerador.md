# Gerador do relatório (.docx)

Monta `relatorio/Relatorio_MLP_ENEM.docx` a partir das figuras e tabelas em `outputs/`.

```bash
cd relatorio/gerador && npm install && GITHUB_URL=https://github.com/PedroAllien/enem-mlp-nota-media node main.js
```

O texto de cada seção está nos arquivos `sec_*.js` e `parte1.js`; os números são lidos de `outputs/tables/`.
Depois de abrir no Word, atualize o sumário (botão direito → Atualizar campo).
