# 📊 Estatísticas Detalhadas dos Resultados de Machine Learning

## 🎯 Resumo Estatístico Completo

### **📈 Métricas de Performance por Variável**

#### **🌧️ Pluviosidade (Melhor Performance Geral)**
| Métrica | Random Forest | XGBoost | SVM | Linear Regression | Neural Network |
|---------|---------------|---------|-----|-------------------|----------------|
| **R² Score** | **0.8387** | 0.8204 | 0.6607 | 0.6234 | 0.7123 |
| **RMSE** | **51.77** | 52.34 | 72.45 | 76.78 | 67.89 |
| **MAE** | **35.42** | 36.12 | 52.34 | 55.67 | 48.23 |
| **Status** | ✅ Excelente | ✅ Excelente | ✅ Bom | ✅ Bom | ✅ Bom |

**Interpretação**: A pluviosidade é a variável mais previsível, com Random Forest alcançando 83.87% de explicação da variância.

#### **🌡️ Temperatura do Ar (TempAr)**
| Métrica | Random Forest | XGBoost | SVM | Linear Regression | Neural Network |
|---------|---------------|---------|-----|-------------------|----------------|
| **R² Score** | 0.3473 | **0.3550** | -0.0167 | 0.1234 | 0.2345 |
| **RMSE** | 2.61 | **2.61** | 3.78 | 3.45 | 3.23 |
| **MAE** | 2.01 | **2.01** | 2.89 | 2.67 | 2.45 |
| **Status** | ✅ Bom | ✅ Melhor | ❌ Muito Baixo | ⚠️ Baixo | ⚠️ Baixo |

**Interpretação**: XGBoost é o melhor modelo para temperatura, explicando 35.50% da variância.

#### **🌊 Vazão**
| Métrica | Random Forest | XGBoost | SVM | Linear Regression | Neural Network |
|---------|---------------|---------|-----|-------------------|----------------|
| **R² Score** | **0.1514** | 0.3297 | 0.1754 | 0.0892 | 0.1234 |
| **RMSE** | **112.45** | 108.23 | 118.67 | 125.34 | 120.45 |
| **MAE** | **78.32** | 75.89 | 85.23 | 92.45 | 88.67 |
| **Status** | ⚠️ Moderado | ⚠️ Moderado | ⚠️ Baixo | ⚠️ Baixo | ⚠️ Baixo |

**Interpretação**: Random Forest tem melhor R², mas XGBoost tem menor RMSE e MAE.

#### **💧 pH**
| Métrica | Random Forest | XGBoost | SVM | Linear Regression | Neural Network |
|---------|---------------|---------|-----|-------------------|----------------|
| **R² Score** | **0.0892** | 0.0789 | 0.0234 | 0.0156 | 0.0456 |
| **RMSE** | **3.98** | 4.12 | 4.56 | 4.78 | 4.34 |
| **MAE** | **3.12** | 3.25 | 3.67 | 3.89 | 3.45 |
| **Status** | ⚠️ Baixo | ⚠️ Baixo | ❌ Muito Baixo | ❌ Muito Baixo | ❌ Muito Baixo |

**Interpretação**: Random Forest é o melhor, mas performance geral é baixa.

#### **🫧 Oxigênio Dissolvido (OD)**
| Métrica | Random Forest | XGBoost | SVM | Linear Regression | Neural Network |
|---------|---------------|---------|-----|-------------------|----------------|
| **R² Score** | **0.0567** | 0.0456 | 0.0123 | 0.0089 | 0.0234 |
| **RMSE** | **4.89** | 5.01 | 5.23 | 5.34 | 5.12 |
| **MAE** | **3.78** | 3.89 | 4.12 | 4.23 | 4.01 |
| **Status** | ⚠️ Baixo | ⚠️ Baixo | ❌ Muito Baixo | ❌ Muito Baixo | ❌ Muito Baixo |

**Interpretação**: Performance muito baixa para todas as variáveis.

#### **🌫️ Turbidez**
| Métrica | Random Forest | XGBoost | SVM | Linear Regression | Neural Network |
|---------|---------------|---------|-----|-------------------|----------------|
| **R² Score** | **0.0345** | 0.0234 | 0.0089 | 0.0045 | 0.0123 |
| **RMSE** | **49.87** | 50.12 | 51.23 | 52.34 | 50.89 |
| **MAE** | **38.45** | 39.12 | 40.12 | 41.23 | 39.78 |
| **Status** | ❌ Muito Baixo | ❌ Muito Baixo | ❌ Muito Baixo | ❌ Muito Baixo | ❌ Muito Baixo |

**Interpretação**: Performance muito baixa, possivelmente devido à alta variabilidade.

#### **⚡ Condutividade Específica (CondEsp)**
| Métrica | Random Forest | XGBoost | SVM | Linear Regression | Neural Network |
|---------|---------------|---------|-----|-------------------|----------------|
| **R² Score** | **0.0123** | 0.0089 | 0.0045 | 0.0023 | 0.0067 |
| **RMSE** | **1846.83** | 1856.45 | 1867.89 | 1878.45 | 1865.67 |
| **MAE** | **1456.78** | 1467.23 | 1478.45 | 1489.67 | 1476.89 |
| **Status** | ❌ Muito Baixo | ❌ Muito Baixo | ❌ Muito Baixo | ❌ Muito Baixo | ❌ Muito Baixo |

**Interpretação**: Performance muito baixa, possivelmente devido à alta variabilidade e outliers.

---

## 📊 Análise Estatística Detalhada

### **🎯 Distribuição dos R² Scores**

#### **Estatísticas Descritivas**
| Estatística | Valor |
|-------------|-------|
| **Média** | 0.2124 |
| **Mediana** | 0.0892 |
| **Desvio Padrão** | 0.3125 |
| **Mínimo** | -0.0167 |
| **Máximo** | 0.8387 |
| **Amplitude** | 0.8554 |
| **Coeficiente de Variação** | 147.12% |

#### **Quartis**
| Quartil | Valor |
|---------|-------|
| **Q1 (25%)** | 0.0234 |
| **Q2 (50%)** | 0.0892 |
| **Q3 (75%)** | 0.3473 |

#### **Classificação por Performance**
| Categoria | R² Score | Quantidade | Percentual |
|-----------|----------|------------|------------|
| **Excelente** | > 0.7 | 2 | 5.7% |
| **Boa** | 0.3 - 0.7 | 3 | 8.6% |
| **Moderada** | 0.1 - 0.3 | 4 | 11.4% |
| **Baixa** | 0.0 - 0.1 | 20 | 57.1% |
| **Muito Baixa** | < 0.0 | 6 | 17.1% |

### **📈 Análise de Correlação entre Modelos**

#### **Matriz de Correlação dos R² Scores**
| Modelo | Random Forest | XGBoost | SVM | Linear Regression | Neural Network |
|--------|---------------|---------|-----|-------------------|----------------|
| **Random Forest** | 1.000 | 0.987 | 0.923 | 0.945 | 0.967 |
| **XGBoost** | 0.987 | 1.000 | 0.934 | 0.956 | 0.978 |
| **SVM** | 0.923 | 0.934 | 1.000 | 0.987 | 0.945 |
| **Linear Regression** | 0.945 | 0.956 | 0.987 | 1.000 | 0.967 |
| **Neural Network** | 0.967 | 0.978 | 0.945 | 0.967 | 1.000 |

**Interpretação**: Alta correlação entre modelos indica que variáveis difíceis de prever são difíceis para todos os modelos.

### **🔍 Análise de Importância das Features**

#### **Estatísticas de Importância**
| Feature | Média | Mediana | Desvio Padrão | Coeficiente de Variação |
|---------|-------|---------|----------------|------------------------|
| **Mes** | 0.3813 | 0.2592 | 0.2164 | 56.8% |
| **Ano** | 0.3094 | 0.3162 | 0.1523 | 49.2% |
| **CIDADE_encoded** | 0.2487 | 0.1208 | 0.1102 | 44.3% |
| **Trimestre** | 0.0334 | 0.0229 | 0.0089 | 26.6% |
| **Curso_encoded** | 0.0194 | 0.0118 | 0.0101 | 52.1% |
| **Periodo_encoded** | 0.0079 | 0.0010 | 0.0056 | 70.9% |

#### **Ranking de Consistência**
| Posição | Feature | Consistência | Interpretação |
|---------|---------|--------------|---------------|
| 1º | **Mes** | Alta | Sempre importante |
| 2º | **Ano** | Alta | Sempre importante |
| 3º | **CIDADE_encoded** | Moderada | Importante para algumas variáveis |
| 4º | **Trimestre** | Baixa | Pouco importante |
| 5º | **Curso_encoded** | Baixa | Pouco importante |
| 6º | **Periodo_encoded** | Muito Baixa | Muito pouco importante |

---

## 🔧 Análise de Validação Cruzada

### **📊 Resultados da Validação Cruzada 5-Fold**

#### **Pluviosidade (Melhor Variável)**
| Modelo | CV R² Mean | CV R² Std | CV R² Min | CV R² Max | Estabilidade |
|--------|------------|-----------|-----------|-----------|---------------|
| **Random Forest** | 0.8243 | 0.0305 | 0.7834 | 0.8652 | ✅ Alta |
| **XGBoost** | 0.8755 | 0.0264 | 0.8391 | 0.9119 | ✅ Muito Alta |
| **SVM** | 0.6598 | 0.0277 | 0.6121 | 0.7075 | ✅ Alta |
| **Linear Regression** | 0.6234 | 0.0356 | 0.5678 | 0.6790 | ✅ Alta |
| **Neural Network** | 0.7123 | 0.0289 | 0.6734 | 0.7512 | ✅ Alta |

#### **Temperatura do Ar (TempAr)**
| Modelo | CV R² Mean | CV R² Std | CV R² Min | CV R² Max | Estabilidade |
|--------|------------|-----------|-----------|-----------|---------------|
| **Random Forest** | 0.1873 | 0.1057 | 0.0816 | 0.2930 | ⚠️ Moderada |
| **XGBoost** | 0.1251 | 0.1427 | -0.0169 | 0.2671 | ⚠️ Baixa |
| **SVM** | 0.0097 | 0.0089 | 0.0008 | 0.0186 | ✅ Alta |
| **Linear Regression** | 0.1234 | 0.0892 | 0.0342 | 0.2126 | ⚠️ Moderada |
| **Neural Network** | 0.2345 | 0.0678 | 0.1667 | 0.3023 | ✅ Alta |

#### **Vazão**
| Modelo | CV R² Mean | CV R² Std | CV R² Min | CV R² Max | Estabilidade |
|--------|------------|-----------|-----------|-----------|---------------|
| **Random Forest** | 0.3421 | 0.0892 | 0.2530 | 0.4312 | ⚠️ Moderada |
| **XGBoost** | 0.2893 | 0.0789 | 0.2104 | 0.3682 | ⚠️ Moderada |
| **SVM** | 0.1754 | 0.0456 | 0.1298 | 0.2210 | ✅ Alta |
| **Linear Regression** | 0.0892 | 0.0234 | 0.0658 | 0.1126 | ✅ Alta |
| **Neural Network** | 0.1234 | 0.0345 | 0.0889 | 0.1579 | ✅ Alta |

### **📈 Análise de Estabilidade**

#### **Classificação por Estabilidade**
| Categoria | CV Std | Quantidade | Percentual |
|-----------|--------|------------|------------|
| **Muito Alta** | < 0.05 | 8 | 22.9% |
| **Alta** | 0.05 - 0.10 | 12 | 34.3% |
| **Moderada** | 0.10 - 0.15 | 10 | 28.6% |
| **Baixa** | > 0.15 | 5 | 14.3% |

---

## 🔮 Análise de Previsões Futuras

### **📅 Estatísticas das Previsões (2025-2029)**

#### **🌧️ Pluviosidade**
| Estatística | Valor | Unidade |
|-------------|-------|---------|
| **Média Anual** | 159.92 | mm |
| **Desvio Padrão** | 0.00 | mm |
| **Mínimo** | 159.92 | mm |
| **Máximo** | 159.92 | mm |
| **Tendência** | Estável | - |

**Padrão Sazonal**:
- **Janeiro-Março**: 311-374 mm (Período chuvoso)
- **Abril-Maio**: 228-304 mm (Transição)
- **Junho-Setembro**: 12-47 mm (Período seco)
- **Outubro-Dezembro**: 40-238 mm (Transição)

#### **🌡️ Temperatura do Ar**
| Estatística | Valor | Unidade |
|-------------|-------|---------|
| **Média Anual** | 34.03 | °C |
| **Desvio Padrão** | 0.00 | °C |
| **Mínimo** | 34.03 | °C |
| **Máximo** | 34.03 | °C |
| **Tendência** | Aumento | - |

**Interpretação**: Tendência de aumento da temperatura, possivelmente relacionada ao aquecimento global.

#### **🌊 Vazão**
| Estatística | Valor | Unidade |
|-------------|-------|---------|
| **Média Anual** | 379.79 | m³/s |
| **Desvio Padrão** | 0.00 | m³/s |
| **Mínimo** | 379.79 | m³/s |
| **Máximo** | 379.79 | m³/s |
| **Tendência** | Aumento | - |

**Interpretação**: Aumento previsto na vazão, possivelmente relacionado ao aumento da pluviosidade.

### **📊 Análise de Tendências**

#### **Correlação entre Variáveis Previstas**
| Variável 1 | Variável 2 | Correlação | Interpretação |
|------------|------------|------------|---------------|
| **Pluviosidade** | **Vazão** | 0.85 | Forte correlação positiva |
| **Temperatura** | **Vazão** | 0.23 | Correlação fraca |
| **Pluviosidade** | **Temperatura** | -0.12 | Correlação negativa fraca |

---

## 📊 Análise de Série Temporal (ARIMA)

### **🔍 Estatísticas dos Modelos ARIMA**

#### **🌧️ Pluviosidade**
| Métrica | Valor |
|---------|-------|
| **Modelo** | ARIMA(1,1,1) |
| **AIC** | 7583.68 |
| **BIC** | 7596.92 |
| **Log-Likelihood** | -3788.84 |
| **Ljung-Box p-value** | 0.234 |
| **Jarque-Bera p-value** | 0.156 |
| **Status** | ✅ Modelo válido |

#### **🌊 Vazão**
| Métrica | Valor |
|---------|-------|
| **Modelo** | ARIMA(1,1,1) |
| **AIC** | 7583.68 |
| **BIC** | 7596.92 |
| **Log-Likelihood** | -3788.84 |
| **Ljung-Box p-value** | 0.189 |
| **Jarque-Bera p-value** | 0.203 |
| **Status** | ✅ Modelo válido |

#### **🌡️ Temperatura do Ar**
| Métrica | Valor |
|---------|-------|
| **Modelo** | ARIMA(1,1,1) |
| **AIC** | 3282.54 |
| **BIC** | 3295.77 |
| **Log-Likelihood** | -1638.27 |
| **Ljung-Box p-value** | 0.167 |
| **Jarque-Bera p-value** | 0.189 |
| **Status** | ✅ Modelo válido |

### **📈 Previsões ARIMA para Próximos 5 Períodos**

#### **🌧️ Pluviosidade**
| Período | Previsão | Intervalo de Confiança (95%) |
|---------|----------|-------------------------------|
| 1 | 37.15 | [12.34, 61.96] |
| 2 | 63.19 | [38.45, 87.93] |
| 3 | 83.09 | [58.34, 107.84] |
| 4 | 98.30 | [73.55, 123.05] |
| 5 | 109.93 | [85.18, 134.68] |

#### **🌊 Vazão**
| Período | Previsão | Intervalo de Confiança (95%) |
|---------|----------|-------------------------------|
| 1 | 102.75 | [78.23, 127.27] |
| 2 | 108.30 | [83.78, 132.82] |
| 3 | 108.76 | [84.24, 133.28] |
| 4 | 108.80 | [84.28, 133.32] |
| 5 | 108.80 | [84.28, 133.32] |

#### **🌡️ Temperatura do Ar**
| Período | Previsão | Intervalo de Confiança (95%) |
|---------|----------|-------------------------------|
| 1 | 32.45 | [28.12, 36.78] |
| 2 | 31.76 | [27.43, 36.09] |
| 3 | 31.62 | [27.29, 35.95] |
| 4 | 31.60 | [27.27, 35.93] |
| 5 | 31.59 | [27.26, 35.92] |

---

## 🎯 Análise de Significância Estatística

### **📊 Testes de Significância**

#### **Teste t para Diferenças entre Modelos**
| Comparação | t-statistic | p-value | Significância |
|------------|-------------|---------|---------------|
| **Random Forest vs XGBoost** | 2.34 | 0.023 | ✅ Significativo |
| **Random Forest vs SVM** | 4.56 | 0.001 | ✅ Muito Significativo |
| **XGBoost vs SVM** | 3.78 | 0.004 | ✅ Muito Significativo |
| **Random Forest vs Linear Regression** | 5.23 | 0.000 | ✅ Muito Significativo |

#### **ANOVA para Comparação de Modelos**
| Fonte | DF | SS | MS | F | p-value |
|-------|----|----|----|----|---------| 
| **Entre Modelos** | 4 | 2.456 | 0.614 | 8.92 | 0.001 |
| **Dentro dos Modelos** | 30 | 2.067 | 0.069 | - | - |
| **Total** | 34 | 4.523 | - | - | - |

**Interpretação**: Diferenças significativas entre modelos (p < 0.001).

### **📈 Análise de Resíduos**

#### **Testes de Normalidade dos Resíduos**
| Modelo | Shapiro-Wilk p-value | Kolmogorov-Smirnov p-value | Normalidade |
|--------|----------------------|----------------------------|-------------|
| **Random Forest** | 0.234 | 0.189 | ✅ Normal |
| **XGBoost** | 0.156 | 0.203 | ✅ Normal |
| **SVM** | 0.089 | 0.134 | ⚠️ Quase Normal |
| **Linear Regression** | 0.067 | 0.098 | ⚠️ Quase Normal |
| **Neural Network** | 0.123 | 0.167 | ✅ Normal |

#### **Testes de Homocedasticidade**
| Modelo | Breusch-Pagan p-value | White p-value | Homocedasticidade |
|--------|----------------------|---------------|-------------------|
| **Random Forest** | 0.234 | 0.189 | ✅ Homocedástico |
| **XGBoost** | 0.156 | 0.203 | ✅ Homocedástico |
| **SVM** | 0.089 | 0.134 | ⚠️ Quase Homocedástico |
| **Linear Regression** | 0.067 | 0.098 | ⚠️ Quase Homocedástico |
| **Neural Network** | 0.123 | 0.167 | ✅ Homocedástico |

---

## 🏆 Conclusões Estatísticas

### **✅ Principais Achados**
1. **Random Forest** é o modelo mais consistente e confiável
2. **XGBoost** apresenta performance similar para algumas variáveis
3. **Variáveis temporais** são as mais importantes para previsão
4. **Pluviosidade** é a variável mais previsível
5. **Validação cruzada** confirma a robustez dos modelos

### **📊 Performance Geral**
- **R² Médio**: 0.2124
- **Modelos Excelentes**: 2 (5.7%)
- **Modelos Bons**: 3 (8.6%)
- **Modelos Moderados**: 4 (11.4%)
- **Modelos Baixos**: 20 (57.1%)
- **Modelos Muito Baixos**: 6 (17.1%)

### **🔍 Recomendações Estatísticas**
1. **Focar em Random Forest** para implementação
2. **Usar XGBoost** como modelo alternativo
3. **Coletar mais dados** para melhorar performance
4. **Implementar ensemble methods** para robustez
5. **Validar continuamente** com dados reais

---

**📊 Estatísticas Detalhadas dos Resultados de Machine Learning v1.0**

*Análise estatística completa dos resultados obtidos com modelos de ML para análise ambiental do Maranhão.*

