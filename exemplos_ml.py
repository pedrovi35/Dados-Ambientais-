# 💻 Exemplos Práticos de Código - Machine Learning

## 🌿 Implementação Completa dos Modelos

### **1. Função Principal de Treinamento**

```python
def train_ml_models(df):
    """
    Treina modelos de machine learning para análise ambiental
    
    Args:
        df (pd.DataFrame): DataFrame com dados ambientais
        
    Returns:
        tuple: (results, predictions, y_test, models, scaler, feature_names)
    """
    
    # Preparar dados para ML
    variaveis_ml = ['Pluviosidade', 'Vazão', 'TempAr', 'pH', 'OD', 'Turbidez', 'CondEsp']
    variaveis_features = ['Ano', 'Mes', 'Trimestre', 'CIDADE_encoded', 'Periodo_encoded', 'Curso_encoded']
    
    # Filtrar dados válidos
    df_ml = df[variaveis_ml + variaveis_features].dropna()
    
    if len(df_ml) < 10:
        return None
    
    # Selecionar variável para previsão (pH como exemplo)
    target_var = 'pH'
    y = df_ml[target_var].dropna()
    X = df_ml.loc[y.index, variaveis_features]
    
    # Dividir dados
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42
    )
    
    # Normalizar features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Definir modelos
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'SVM': SVR(kernel='rbf', C=1.0, gamma='scale'),
        'Linear Regression': LinearRegression(),
        'Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    }
    
    # Treinar modelos e calcular métricas
    results = {}
    predictions = {}
    
    for name, model in models.items():
        try:
            if name in ['SVM', 'Linear Regression', 'Neural Network']:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Calcular métricas
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            results[name] = {'R²': r2, 'RMSE': rmse, 'MAE': mae}
            predictions[name] = y_pred
            
        except Exception as e:
            print(f"Erro ao treinar {name}: {e}")
    
    return results, predictions, y_test, models, scaler, X.columns
```

### **2. Função de Otimização de Hiperparâmetros**

```python
def otimizar_hiperparametros(X, y, nome_variavel):
    """
    Otimiza hiperparâmetros dos modelos usando GridSearchCV
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Variável alvo
        nome_variavel (str): Nome da variável sendo analisada
        
    Returns:
        tuple: (resultados_otimizados, cv_resultados)
    """
    
    # Dividir dados
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalizar
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Grid de parâmetros para Random Forest
    param_grid_rf = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Grid de parâmetros para SVM
    param_grid_svm = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
        'epsilon': [0.01, 0.1, 0.2]
    }
    
    # Grid de parâmetros para XGBoost
    param_grid_xgb = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0]
    }
    
    # Executar GridSearchCV
    resultados_otimizados = {}
    cv_resultados = {}
    
    # Random Forest
    grid_rf = GridSearchCV(
        RandomForestRegressor(random_state=42),
        param_grid_rf,
        cv=5,
        scoring='r2',
        n_jobs=-1
    )
    grid_rf.fit(X_train, y_train)
    
    # Treinar modelo otimizado
    rf_otimizado = grid_rf.best_estimator_
    y_pred_rf = rf_otimizado.predict(X_test)
    
    resultados_otimizados['Random Forest'] = {
        'R²': r2_score(y_test, y_pred_rf),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_rf)),
        'MAE': mean_absolute_error(y_test, y_pred_rf),
        'Melhores Parâmetros': grid_rf.best_params_
    }
    
    # Validação cruzada
    cv_scores_rf = cross_val_score(rf_otimizado, X_train, y_train, cv=5, scoring='r2')
    cv_resultados['Random Forest'] = {
        'CV Mean': cv_scores_rf.mean(),
        'CV Std': cv_scores_rf.std()
    }
    
    # SVM
    grid_svm = GridSearchCV(
        SVR(kernel='rbf'),
        param_grid_svm,
        cv=5,
        scoring='r2',
        n_jobs=-1
    )
    grid_svm.fit(X_train_scaled, y_train)
    
    svm_otimizado = grid_svm.best_estimator_
    y_pred_svm = svm_otimizado.predict(X_test_scaled)
    
    resultados_otimizados['SVM'] = {
        'R²': r2_score(y_test, y_pred_svm),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_svm)),
        'MAE': mean_absolute_error(y_test, y_pred_svm),
        'Melhores Parâmetros': grid_svm.best_params_
    }
    
    cv_scores_svm = cross_val_score(svm_otimizado, X_train_scaled, y_train, cv=5, scoring='r2')
    cv_resultados['SVM'] = {
        'CV Mean': cv_scores_svm.mean(),
        'CV Std': cv_scores_svm.std()
    }
    
    return resultados_otimizados, cv_resultados
```

### **3. Função de Previsões Futuras**

```python
def fazer_previsoes_futuras(variavel, melhor_modelo, X, y):
    """
    Faz previsões futuras para uma variável específica
    
    Args:
        variavel (str): Nome da variável
        melhor_modelo: Modelo treinado
        X (pd.DataFrame): Features
        y (pd.Series): Variável alvo
        
    Returns:
        tuple: (previsoes_futuras, r2_teste, rmse_teste)
    """
    
    # Dividir dados
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalizar
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Treinar modelo final
    if melhor_modelo.__class__.__name__ in ['SVR', 'LinearRegression', 'MLPRegressor']:
        melhor_modelo.fit(X_train_scaled, y_train)
        y_pred_teste = melhor_modelo.predict(X_test_scaled)
    else:
        melhor_modelo.fit(X_train, y_train)
        y_pred_teste = melhor_modelo.predict(X_test)
    
    # Calcular métricas no conjunto de teste
    r2_teste = r2_score(y_test, y_pred_teste)
    rmse_teste = np.sqrt(mean_squared_error(y_test, y_pred_teste))
    
    # Gerar dados futuros (próximos 5 anos)
    anos_futuros = range(2020, 2025)
    meses = range(1, 13)
    
    previsoes_futuras = []
    
    for ano in anos_futuros:
        for mes in meses:
            # Criar features para o futuro
            features_futuras = np.array([[
                ano,
                mes,
                (mes - 1) // 3 + 1,  # Trimestre
                0,  # CIDADE_encoded (assumindo Coroatá)
                1 if mes in [1,2,3,4,5,6] else 0,  # Periodo_encoded
                0   # Curso_encoded (assumindo Baixo)
            ]])
            
            # Fazer previsão
            if melhor_modelo.__class__.__name__ in ['SVR', 'LinearRegression', 'MLPRegressor']:
                features_scaled = scaler.transform(features_futuras)
                previsao = melhor_modelo.predict(features_scaled)[0]
            else:
                previsao = melhor_modelo.predict(features_futuras)[0]
            
            previsoes_futuras.append({
                'Ano': ano,
                'Mes': mes,
                'Previsao': previsao
            })
    
    # Calcular médias anuais
    df_previsoes = pd.DataFrame(previsoes_futuras)
    medias_anuais = df_previsoes.groupby('Ano')['Previsao'].mean()
    
    print(f"\n📊 Previsões Futuras para {variavel}")
    print("="*50)
    print("Médias Anuais Previstas:")
    for ano, media in medias_anuais.items():
        print(f"  {ano}: {media:.3f}")
    
    return previsoes_futuras, r2_teste, rmse_teste
```

### **4. Função de Análise de Importância das Features**

```python
def analisar_importancia_features(variavel, X, y):
    """
    Analisa a importância das features usando Random Forest
    
    Args:
        variavel (str): Nome da variável
        X (pd.DataFrame): Features
        y (pd.Series): Variável alvo
        
    Returns:
        pd.DataFrame: DataFrame com importância das features
    """
    
    # Treinar Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # Obter importância das features
    importancia = rf.feature_importances_
    
    # Criar DataFrame
    df_importancia = pd.DataFrame({
        'Feature': X.columns,
        'Importância': importancia
    }).sort_values('Importância', ascending=False)
    
    print(f"\n🔍 Importância das Features para {variavel}")
    print("="*50)
    for _, row in df_importancia.iterrows():
        print(f"  {row['Feature']}: {row['Importância']:.3f}")
    
    return df_importancia
```

### **5. Função de Visualização dos Resultados**

```python
def criar_visualizacoes_ml(results, predictions, y_test, df_importancia):
    """
    Cria visualizações para os resultados de ML
    
    Args:
        results (dict): Resultados dos modelos
        predictions (dict): Previsões dos modelos
        y_test (pd.Series): Valores reais de teste
        df_importancia (pd.DataFrame): Importância das features
    """
    
    # 1. Comparação de performance
    df_results = pd.DataFrame(results).T
    df_results = df_results.sort_values('R²', ascending=False)
    
    fig_performance = px.bar(
        df_results.reset_index(),
        x='index',
        y='R²',
        title='Performance dos Modelos de ML (R² Score)',
        labels={'index': 'Modelo', 'R²': 'R² Score'},
        color='R²',
        color_continuous_scale='viridis',
        template='plotly_dark'
    )
    fig_performance.update_layout(height=400)
    
    # 2. Previsões vs Valores Reais
    fig_predictions = make_subplots(
        rows=2, cols=2,
        subplot_titles=list(predictions.keys()),
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )
    
    for i, (name, pred) in enumerate(predictions.items()):
        row = (i // 2) + 1
        col = (i % 2) + 1
        
        fig_predictions.add_trace(
            go.Scatter(
                x=y_test,
                y=pred,
                mode='markers',
                name=name,
                showlegend=False
            ),
            row=row, col=col
        )
        
        # Adicionar linha de referência
        min_val = min(y_test.min(), pred.min())
        max_val = max(y_test.max(), pred.max())
        fig_predictions.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                line=dict(dash='dash', color='red'),
                name='Linha Perfeita',
                showlegend=False
            ),
            row=row, col=col
        )
    
    fig_predictions.update_layout(height=600, title_text="Previsões vs Valores Reais", template='plotly_dark')
    
    # 3. Importância das features
    fig_importancia = px.bar(
        df_importancia,
        x='Importância',
        y='Feature',
        orientation='h',
        title='Importância das Features (Random Forest)',
        color='Importância',
        color_continuous_scale='viridis',
        template='plotly_dark'
    )
    fig_importancia.update_layout(height=400)
    
    return fig_performance, fig_predictions, fig_importancia
```

### **6. Exemplo de Uso Completo**

```python
# Exemplo de uso completo do sistema de ML
def exemplo_completo_ml():
    """
    Exemplo completo de uso do sistema de Machine Learning
    """
    
    # Carregar dados
    df = pd.read_csv('dadosmestrado.csv', sep=';', decimal=',')
    
    # Preparar dados
    df.columns = df.columns.str.strip()
    df['DATA'] = pd.to_datetime(df['DATA'], format='%d/%m/%Y')
    df['Ano'] = df['DATA'].dt.year
    df['Mes'] = df['DATA'].dt.month
    df['Trimestre'] = df['DATA'].dt.quarter
    
    # Converter colunas numéricas
    colunas_numericas = ['Pluviosidade', 'Vazão', 'TempAr', 'TempAmostra', 'pH', 
                         'SolsuspTot', 'SoldissTot', 'Turbidez', 'CondEle', 'OD', 'CondEsp']
    
    for col in colunas_numericas:
        if col in df.columns:
            df[col] = df[col].astype(str)
            df[col] = df[col].str.replace(',', '.').str.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Criar variáveis categóricas
    df['Periodo'] = df['Mes'].apply(lambda x: 'Chuvoso' if x in [1,2,3,4,5,6] else 'Estiagem')
    df['Curso'] = df['CIDADE'].apply(lambda x: 'Baixo' if 'Coroatá' in x else 'Médio')
    
    # Codificar variáveis categóricas
    le_cidade = LabelEncoder()
    le_periodo = LabelEncoder()
    le_curso = LabelEncoder()
    
    df['CIDADE_encoded'] = le_cidade.fit_transform(df['CIDADE'])
    df['Periodo_encoded'] = le_periodo.fit_transform(df['Periodo'])
    df['Curso_encoded'] = le_curso.fit_transform(df['Curso'])
    
    # Selecionar variável para análise
    variavel_alvo = 'pH'
    variaveis_features = ['Ano', 'Mes', 'Trimestre', 'CIDADE_encoded', 'Periodo_encoded', 'Curso_encoded']
    
    # Preparar dados para ML
    df_ml = df[variaveis_features + [variavel_alvo]].dropna()
    X = df_ml[variaveis_features]
    y = df_ml[variavel_alvo]
    
    print(f"📊 Análise de Machine Learning para {variavel_alvo}")
    print("="*60)
    print(f"Dados disponíveis: {len(df_ml)} registros")
    print(f"Features: {len(variaveis_features)}")
    
    # 1. Treinar modelos básicos
    print("\n🤖 Treinando modelos básicos...")
    results, predictions, y_test, models, scaler, feature_names = train_ml_models(df)
    
    if results:
        # Mostrar resultados
        df_results = pd.DataFrame(results).T
        df_results = df_results.sort_values('R²', ascending=False)
        print("\n📈 Resultados dos Modelos:")
        print(df_results.round(4))
        
        # 2. Otimizar hiperparâmetros
        print("\n🔧 Otimizando hiperparâmetros...")
        resultados_otimizados, cv_resultados = otimizar_hiperparametros(X, y, variavel_alvo)
        
        # 3. Análise de importância
        print("\n🔍 Analisando importância das features...")
        df_importancia = analisar_importancia_features(variavel_alvo, X, y)
        
        # 4. Previsões futuras
        print("\n🔮 Fazendo previsões futuras...")
        melhor_modelo = models[df_results.index[0]]  # Melhor modelo
        previsoes_futuras, r2_teste, rmse_teste = fazer_previsoes_futuras(
            variavel_alvo, melhor_modelo, X, y
        )
        
        # 5. Criar visualizações
        print("\n📊 Criando visualizações...")
        fig_performance, fig_predictions, fig_importancia = criar_visualizacoes_ml(
            results, predictions, y_test, df_importancia
        )
        
        print("\n✅ Análise de Machine Learning concluída!")
        print(f"Melhor modelo: {df_results.index[0]}")
        print(f"R² Score: {df_results.iloc[0]['R²']:.3f}")
        print(f"RMSE: {df_results.iloc[0]['RMSE']:.3f}")
        
        return {
            'results': results,
            'predictions': predictions,
            'y_test': y_test,
            'models': models,
            'scaler': scaler,
            'feature_importance': df_importancia,
            'future_predictions': previsoes_futuras,
            'visualizations': {
                'performance': fig_performance,
                'predictions': fig_predictions,
                'importance': fig_importancia
            }
        }
    
    else:
        print("❌ Dados insuficientes para treinamento dos modelos.")
        return None

# Executar exemplo
if __name__ == "__main__":
    resultado = exemplo_completo_ml()
```

---

## 🎯 Casos de Uso Específicos

### **1. Análise de Qualidade da Água**

```python
def analisar_qualidade_agua(df):
    """
    Análise específica para qualidade da água usando pH
    """
    
    # Preparar dados para pH
    variaveis_ph = ['Ano', 'Mes', 'Trimestre', 'CIDADE_encoded', 'Periodo_encoded', 'Curso_encoded']
    df_ph = df[variaveis_ph + ['pH']].dropna()
    
    X = df_ph[variaveis_ph]
    y = df_ph['pH']
    
    # Treinar modelos
    results, predictions, y_test, models, scaler, _ = train_ml_models(df)
    
    # Análise específica para pH
    print("🌊 Análise de Qualidade da Água (pH)")
    print("="*40)
    
    # Verificar se pH está dentro dos padrões
    ph_medio = y.mean()
    ph_std = y.std()
    
    print(f"pH Médio: {ph_medio:.2f}")
    print(f"Desvio Padrão: {ph_std:.2f}")
    
    if 6.0 <= ph_medio <= 9.0:
        print("✅ pH dentro dos padrões de qualidade (6.0-9.0)")
    else:
        print("⚠️ pH fora dos padrões de qualidade")
    
    return results, predictions, y_test, models, scaler
```

### **2. Previsão de Vazão**

```python
def prever_vazao(df):
    """
    Previsão específica para vazão do rio
    """
    
    # Preparar dados para vazão
    variaveis_vazao = ['Ano', 'Mes', 'Trimestre', 'CIDADE_encoded', 'Periodo_encoded', 'Curso_encoded', 'Pluviosidade']
    df_vazao = df[variaveis_vazao + ['Vazão']].dropna()
    
    X = df_vazao[variaveis_vazao]
    y = df_vazao['Vazão']
    
    # Treinar modelos
    results, predictions, y_test, models, scaler, _ = train_ml_models(df)
    
    print("🌊 Previsão de Vazão")
    print("="*30)
    
    # Análise de correlação com pluviosidade
    correlacao = df_vazao['Pluviosidade'].corr(df_vazao['Vazão'])
    print(f"Correlação Pluviosidade-Vazão: {correlacao:.3f}")
    
    if correlacao > 0.5:
        print("✅ Forte correlação positiva com pluviosidade")
    elif correlacao > 0.3:
        print("📊 Correlação moderada com pluviosidade")
    else:
        print("⚠️ Baixa correlação com pluviosidade")
    
    return results, predictions, y_test, models, scaler
```

### **3. Monitoramento de Temperatura**

```python
def monitorar_temperatura(df):
    """
    Monitoramento e previsão de temperatura
    """
    
    # Preparar dados para temperatura
    variaveis_temp = ['Ano', 'Mes', 'Trimestre', 'CIDADE_encoded', 'Periodo_encoded', 'Curso_encoded']
    df_temp = df[variaveis_temp + ['TempAr']].dropna()
    
    X = df_temp[variaveis_temp]
    y = df_temp['TempAr']
    
    # Treinar modelos
    results, predictions, y_test, models, scaler, _ = train_ml_models(df)
    
    print("🌡️ Monitoramento de Temperatura")
    print("="*35)
    
    # Análise de tendência temporal
    temp_por_ano = df_temp.groupby('Ano')['TempAr'].mean()
    tendencia = temp_por_ano.pct_change().mean()
    
    print(f"Temperatura Média: {y.mean():.1f}°C")
    print(f"Tendência Anual: {tendencia:.3f}°C/ano")
    
    if tendencia > 0:
        print("📈 Tendência de aumento da temperatura")
    elif tendencia < -0.01:
        print("📉 Tendência de diminuição da temperatura")
    else:
        print("📊 Temperatura estável")
    
    return results, predictions, y_test, models, scaler
```

---

## 🔧 Utilitários e Funções Auxiliares

### **1. Função de Validação de Dados**

```python
def validar_dados_ml(df):
    """
    Valida se os dados estão prontos para ML
    
    Args:
        df (pd.DataFrame): DataFrame para validação
        
    Returns:
        bool: True se dados são válidos
    """
    
    # Verificar se há dados suficientes
    if len(df) < 10:
        print("❌ Dados insuficientes (< 10 registros)")
        return False
    
    # Verificar se há variáveis numéricas
    colunas_numericas = df.select_dtypes(include=[np.number]).columns
    if len(colunas_numericas) == 0:
        print("❌ Nenhuma variável numérica encontrada")
        return False
    
    # Verificar valores nulos
    nulos_por_coluna = df.isnull().sum()
    colunas_com_nulos = nulos_por_coluna[nulos_por_coluna > 0]
    
    if len(colunas_com_nulos) > 0:
        print("⚠️ Colunas com valores nulos:")
        for col, nulos in colunas_com_nulos.items():
            percentual = (nulos / len(df)) * 100
            print(f"  {col}: {nulos} nulos ({percentual:.1f}%)")
    
    print("✅ Dados válidos para Machine Learning")
    return True
```

### **2. Função de Salvamento de Modelos**

```python
import joblib

def salvar_modelos(models, scaler, feature_names, nome_arquivo):
    """
    Salva modelos treinados e scaler
    
    Args:
        models (dict): Dicionário com modelos
        scaler: Scaler treinado
        feature_names: Nomes das features
        nome_arquivo (str): Nome do arquivo para salvar
    """
    
    modelo_data = {
        'models': models,
        'scaler': scaler,
        'feature_names': feature_names,
        'timestamp': pd.Timestamp.now()
    }
    
    joblib.dump(modelo_data, f"{nome_arquivo}.pkl")
    print(f"✅ Modelos salvos em {nome_arquivo}.pkl")

def carregar_modelos(nome_arquivo):
    """
    Carrega modelos salvos
    
    Args:
        nome_arquivo (str): Nome do arquivo
        
    Returns:
        dict: Dados dos modelos
    """
    
    modelo_data = joblib.load(f"{nome_arquivo}.pkl")
    print(f"✅ Modelos carregados de {nome_arquivo}.pkl")
    return modelo_data
```

### **3. Função de Relatório de Performance**

```python
def gerar_relatorio_ml(results, predictions, y_test, df_importancia):
    """
    Gera relatório completo de performance dos modelos
    
    Args:
        results (dict): Resultados dos modelos
        predictions (dict): Previsões dos modelos
        y_test (pd.Series): Valores reais de teste
        df_importancia (pd.DataFrame): Importância das features
        
    Returns:
        str: Relatório em texto
    """
    
    relatorio = []
    relatorio.append("📊 RELATÓRIO DE MACHINE LEARNING")
    relatorio.append("="*50)
    
    # Performance dos modelos
    df_results = pd.DataFrame(results).T
    df_results = df_results.sort_values('R²', ascending=False)
    
    relatorio.append("\n🤖 PERFORMANCE DOS MODELOS:")
    relatorio.append("-"*30)
    
    for modelo, metricas in df_results.iterrows():
        relatorio.append(f"{modelo}:")
        relatorio.append(f"  R² Score: {metricas['R²']:.3f}")
        relatorio.append(f"  RMSE: {metricas['RMSE']:.3f}")
        relatorio.append(f"  MAE: {metricas['MAE']:.3f}")
        relatorio.append("")
    
    # Melhor modelo
    melhor_modelo = df_results.index[0]
    relatorio.append(f"🏆 MELHOR MODELO: {melhor_modelo}")
    relatorio.append(f"   R² Score: {df_results.iloc[0]['R²']:.3f}")
    
    # Importância das features
    relatorio.append("\n🔍 IMPORTÂNCIA DAS FEATURES:")
    relatorio.append("-"*30)
    
    for _, row in df_importancia.iterrows():
        relatorio.append(f"{row['Feature']}: {row['Importância']:.3f}")
    
    # Estatísticas dos dados
    relatorio.append("\n📈 ESTATÍSTICAS DOS DADOS:")
    relatorio.append("-"*30)
    relatorio.append(f"Total de registros: {len(y_test)}")
    relatorio.append(f"Valor médio: {y_test.mean():.3f}")
    relatorio.append(f"Desvio padrão: {y_test.std():.3f}")
    relatorio.append(f"Valor mínimo: {y_test.min():.3f}")
    relatorio.append(f"Valor máximo: {y_test.max():.3f}")
    
    return "\n".join(relatorio)
```

---

**🌿 Exemplos Práticos de Código - Machine Learning v1.0**

