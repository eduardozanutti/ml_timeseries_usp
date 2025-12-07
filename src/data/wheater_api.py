# ===================================================================
# 0. Força o script a rodar como se estivesse na raiz do projeto
# ===================================================================
from pathlib import Path
import sys

# Caminho absoluto da raiz do projeto (2 níveis acima)
ROOT = Path(__file__).resolve().parents[2]
# adiciona a raiz ao sys.path
sys.path.append(str(ROOT))

# ===================================================================
# 1. Carregar as Libs
# ===================================================================
import os
from src.config import load_config
import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
import time

# ===================================================================
# 2. Carrega configuração do YAML
# ===================================================================
config = load_config()
print("Configuração carregada com sucesso!")
print(f"Período configurado: {config.get('features',{}).get('weather_features',{}).get('api',{}).get('open-meteo',{}).get('start_date')} até {config.get('features',{}).get('weather_features',{}).get('api',{}).get('open-meteo',{}).get('end_date')}")

# ===================================================================
# 3. Configuração de pastas
# ===================================================================
features = config.get('features', {})

paths = config.get('paths', {})
data_path = paths.get('data', {})
raw_data_path = data_path.get('raw_path', {})

weather_path = os.path.join(raw_data_path, 'weather')
os.makedirs(weather_path, exist_ok=True)
print(f"Pasta de saída criada/verificada: {weather_path}")

# ===================================================================
# 4. Configuração da API Open-Meteo
# ===================================================================
weather_features = features.get('weather_features', {})
api = weather_features.get('api', {}).get('open-meteo', {})

url = api.get('url')  #"https://archive-api.open-meteo.com/v1/archive"
features = api.get('features', [])  # <-- lista de variáveis (ex: ["temperature_2m_mean", ...])
start_date = api.get('start_date', {}).strftime('%Y-%m-%d')
end_date   = api.get('end_date', {}).strftime('%Y-%m-%d')
timezone   = api.get('timezone', {})

print(f"URL da API: {url}")
print(f"Total de variáveis meteorológicas: {len(features)} → {features}")
print(f"Período: {start_date} → {end_date}")
print(f"Timezone: {timezone}")

# ===================================================================
# 5. Carrega coordenadas das lojas
# ===================================================================
coordenates = config.get('coordenates', {})
total_lojas = sum(len(bairros) for cidades in coordenates.values() for bairros in cidades.values())
print(f"Total de lojas encontradas no YAML: {total_lojas}")
print("-" * 60)

# ===================================================================
# 6. Configura cliente com cache + retry
# ===================================================================
cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# ===================================================================
# 7. LOOP PRINCIPAL POR LOJA
# ===================================================================
contador = 0

for state in coordenates.keys():
    for city in coordenates[state].keys():
        for district in coordenates[state][city].keys():
            contador += 1
            
            latitude  = coordenates[state][city][district]['lat']
            longitude = coordenates[state][city][district]['lon']

            print(f"[{contador:3d}/{total_lojas}] Processando → {state} | {city} | {district} ({latitude}, {longitude})")

            params = {
                "latitude": latitude,
                "longitude": longitude,
                "start_date": start_date,
                "end_date": end_date,
                "daily": features,
                "timezone": timezone,
            }
            
            filename = f'{state}-{city}-{district}-weather_features.csv'
            save_path = os.path.join(weather_path, filename)
            
            if os.path.exists(save_path):
                print(f"   Arquivo já existe → pulando: {filename}")
                continue

            # ================================================
            # TRY/EXCEPT
            # ================================================
            try:
                responses = openmeteo.weather_api(url, params=params)
                response = responses[0]
                time.sleep(0.35)

                print(f"   Recebido {response.Daily().TimeEnd() - response.Daily().Time()} dias de dados")

                # ===================================================================
                # 8. Monta DataFrame diário
                # ===================================================================
                daily = response.Daily()

                daily_data = {"date": pd.date_range(
                    start = pd.to_datetime(daily.Time(), unit = "s", utc = True),
                    end =  pd.to_datetime(daily.TimeEnd(), unit = "s", utc = True),
                    freq = pd.Timedelta(seconds = daily.Interval()),
                    inclusive = "left"
                )}

                daily_data['state'] = state
                daily_data['city'] = city
                daily_data['district'] = district
                daily_data['latitude'] = latitude
                daily_data['longitude'] = longitude

                for n, feature in enumerate(features):
                    daily_data[feature] = daily.Variables(n).ValuesAsNumpy()
                
                daily_dataframe = pd.DataFrame(data = daily_data)

                # ===================================================================
                # 9. Agrupa por mês
                # ===================================================================
                daily_dataframe['date'] = daily_dataframe['date'].dt.to_period('M').dt.to_timestamp(how='start')
                
                monthly_dataframe = daily_dataframe.groupby(
                    ['date', 'state', 'city', 'district', 'latitude', 'longitude'],
                    as_index=False
                ).agg({feature: feature.split('_')[-1] for feature in features})

                # ===================================================================
                # 10. Salva CSV
                # ===================================================================
                monthly_dataframe.to_csv(save_path, index=False)
                print(f"Salvo: {save_path}")

            except Exception as e:
                erro_msg = str(e).lower()

                if "minutely api request limit exceeded" in erro_msg or "rate limit" in erro_msg:
                    print(f"   RATE-LIMIT! Dormindo 70 segundos e tentando novamente...")
                    time.sleep(70)
                    continue  # tenta essa mesma loja de novo na próxima iteração

                else:
                    print(f"   ERRO IRRECUPERÁVEL → {state}/{city}/{district}: {e}")
                    continue

print("\nPROCESSAMENTO CONCLUÍDO!")
print(f"Total de arquivos gerados: {contador}")
print(f"Arquivos salvos em: {weather_path}")