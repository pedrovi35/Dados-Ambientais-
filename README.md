


# 🌧️ Dados Ambientais Maranhão - Previsão com Machine Learning

Este repositório contém a análise completa e a implementação de modelos de Machine Learning para previsão de variáveis ambientais críticas no estado do Maranhão, com foco inicial em **Pluviosidade (Chuva)**. O objetivo principal do projeto foi avaliar a capacidade de modelos de regressão em prever variáveis como pluviosidade, vazão e qualidade da água, utilizando dados históricos das estações de monitoramento.

## 📊 Dados

### Fonte dos Dados
Os dados foram consolidados no arquivo `dadosmestrado.csv` e abrangem:

* **Período:** 1992 a 2019 (27 anos de dados mensais).
* **Estações de Monitoramento:** Coroatá e Caxias, entre outras.

### Variáveis Analisadas
O dataset inclui 13 variáveis ambientais principais:

| Variável | Unidade | Descrição |
| :--- | :--- | :--- |
| **Pluviosidade** | mm | Chuva |
| **Vazão** | m³/s | Vazão do Rio |
| **TempAr** | °C | Temperatura do Ar |
| **TempAmostra** | °C | Temperatura da Amostra |
| **pH** | Unidade de pH | Qualidade da Água (Acidez/Alcalinidade) |
| **OD** | mg/L | Oxigênio Dissolvido |
| **Turbidez** | NTU | Turbidez da Água |
| **CondEsp** | μS/cm | Condutividade Específica |
| **MEI** | Índice | Monitoramento Ecológico Integrado |
| **SolsuspTot** | mg/L | Sólidos Suspensos Totais |
| **SoldissTot** | mg/L | Sólidos Dissolvidos Totais |
| **CondEle** | μS/cm | Condutividade Elétrica |
| **ConcentraMatSusp** | mg/L | Concentração de Matéria Suspensa |

## ⚙️ Metodologia

### Pré-processamento
O tratamento dos dados incluiu:
1.  **Limpeza e Conversão:** Tratamento de formatos (`','` para `'.'`) e conversão de colunas para o tipo numérico adequado.
2.  **Tratamento de Nulos:** Utilização de **interpolação linear** para variáveis com menos de 50% de valores nulos (como Pluviosidade, Vazão, TempAr, etc.). Variáveis com alta porcentagem de nulos (SolsuspTot: 87.2%, SoldissTot: 91.0%, ConcentraMatSusp: 69.5%) foram mantidas no DataFrame, mas podem ter sido excluídas em modelos específicos.
3.  **Criação de Features:** Codificação de variáveis categóricas (como **Período** - Chuvoso/Estiagem - e **Curso** - Baixo/Médio) para uso nos modelos de ML.

### Modelos de Machine Learning
Foram implementados e comparados diversos modelos de regressão:

* **Random Forest Regressor**
* **XGBoost**
* **Support Vector Machine (SVM)**
* **Linear Regression**
* **Neural Networks**

## 🏆 Resultados Principais (Previsão - R² Score)

O modelo de Random Forest e XGBoost se destacou significativamente na previsão de Pluviosidade, mas a performance foi muito ruim para a maioria das variáveis de qualidade da água.

| Variável | Melhor Modelo | R² Score | Status |
| :--- | :--- | :--- | :--- |
| **Pluviosidade** (Chuva) | Random Forest | **83.87%** | 🏆 **EXCELENTE** |
| **Temperatura do Ar** | XGBoost | 35.50% | ⚠️ **RUIM** |
| **Vazão** | XGBoost | 32.97% | ⚠️ **RUIM** |
| **pH** | Random Forest | 8.92% | ❌ **MUITO RUIM** |
| **Oxigênio Dissolvido** (OD) | Random Forest | 5.67% | ❌ **MUITO RUIM** |
| **Turbidez** | Random Forest | 3.45% | ❌ **MUITO RUIM** |

### Conclusão e Recomendação
O modelo é **EXCELENTE** para prever a Pluviosidade devido aos padrões sazonais claros e dados consistentes. As previsões futuras (2025-2029) para Pluviosidade indicam uma **Média Anual de 159.92 mm** com forte padrão sazonal (Jan-Mar: 311-374 mm; Jun-Set: 12-47 mm).

A previsão para a qualidade da água e vazão foi baixa, sugerindo que as variáveis de entrada não são suficientes para modelar sua alta variabilidade e as múltiplas influências externas (poluição, atividades humanas).

**Recomendação:**
* **IMPLEMENTAR** o sistema de previsão de chuva imediatamente para suporte a planejamento agrícola e alertas.
* **MELHORAR** os modelos para as outras variáveis, explorando séries temporais (como os modelos ARIMA testados em `dados_possiveis.ipynb`) ou integrando novas *features* de poluição e uso do solo.

## 📂 Estrutura do Repositório

| Arquivo/Pasta | Descrição |
| :--- | :--- |
| `dadosmestrado.csv` | **Dataset** principal com os dados ambientais históricos (1992-2019). |
| `dados.ipynb` | Notebook de **Análise Exploratória de Dados (EDA)**, limpeza e pré-processamento. |
| `dados_possiveis.ipynb` | Notebook para modelagem de **Séries Temporais (ARIMA)**. |
| `documentacao_ml.md` | Documentação detalhada sobre a metodologia e o pré-processamento dos dados. |
| `estatisticas_ml.md` | Detalhamento das métricas de performance (R², RMSE, MAE) de todos os modelos testados. |
| `resumo_ml.md` | Resumo executivo dos resultados, incluindo as conclusões e o *ranking* de performance. |
| `exemplos_ml.py` | Scripts Python contendo exemplos de implementação dos modelos de Machine Learning. |
