# Assistent de tràmits i ajudes · Barcelona

Cas 2 de la microcredencial *Eines i casos d'ús de la IA per a l'Administració Pública* (BSC AI Factory · UPC School).

Pàgina web feta amb **Streamlit** que respon preguntes de la ciutadania sobre tràmits i ajudes de l'**Ajuntament de Barcelona** i de la **Generalitat**, **només a partir dels documents oficials** de la carpeta `dades/`, i en **cita la font**. És el mateix sistema que el quadern del cas 2 (RAG):

1. **Índex:** els PDF es trossegen (1.000 caràcters, 150 de solapament) i es desen a **Chroma** amb els embeddings `text-embedding-3-large`. Es fa sol el primer cop que s'obre l'app (uns 30 segons i uns 3 cèntims).
2. **Cercador MMR:** per a cada pregunta, els 4 trossos més útils i diferents entre ells.
3. **Model de llenguatge** (`gpt-6-luna` d'OpenAI): respon només amb aquests trossos, amb les mateixes 8 normes del quadern, i acaba amb «Font: [n]».
4. **Avís d'IA** visible (art. 50 del Reglament d'IA), botó **«Vull parlar amb una persona»** (010) i **límit de 20 preguntes** per sessió.

> Prototip docent. Els documents són públics i es van consultar l'octubre de 2026: abans de cada edició del curs cal comprovar que continuen vigents.

## Fitxers
| Fitxer | Per a què serveix |
|---|---|
| `app.py` | L'aplicació |
| `requirements.txt` | Llibreries que instal·la el servidor |
| `dades/` | Els PDF oficials i `cas2_documents.csv` (títol, administració i data de cada document) |
| `.streamlit/config.toml` | Colors de la web |
| `.streamlit/secrets.toml.example` | Model del fitxer de secrets (la clau **no** es puja mai a GitHub) |
| `.gitignore` | Evita pujar la clau i l'índex de Chroma per error |

## Publicar-la (GitHub + Streamlit Community Cloud), uns 10 minuts
1. **GitHub:** crea un repositori nou (p. ex. `assistent-tramits-bcn`) i puja-hi tota aquesta carpeta, **excepte** cap fitxer `secrets.toml`.
2. Entra a **https://share.streamlit.io** amb el teu compte de GitHub → **Create app** → tria el repositori, la branca `main` i el fitxer `app.py`.
3. A **Advanced settings → Secrets**, enganxa:
   ```toml
   OPENAI_API_KEY = "sk-..."
   OPENAI_MODEL = "gpt-6-luna"
   ```
4. **Deploy.** En uns minuts tindràs una adreça del tipus `https://assistent-tramits-bcn.streamlit.app`.
5. **Control de despesa:** a la web d'OpenAI, posa un **límit de despesa** al projecte de la clau de classe.

Cada vegada que canviïs un fitxer a GitHub, l'app s'actualitza sola. Si afegeixes o treus PDF, esborra la carpeta `index_chroma` (o reinicia l'app a Streamlit Cloud) perquè es torni a crear l'índex.

## Provar-la al teu ordinador (opcional)
```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # i hi poses la clau
streamlit run app.py
```

## Notes
- **Per què no Vercel:** Vercel executa funcions curtes i sense estat; Streamlit necessita un servidor sempre en marxa. Streamlit Community Cloud és gratuït i publica directament des de GitHub.
- **sqlite3:** Chroma necessita una versió recent de sqlite. A Streamlit Cloud (Linux) s'instal·la `pysqlite3-binary` i l'app el fa servir automàticament; a Windows i macOS no cal.
- **Preguntes de seguiment:** si la pregunta és curta (*«I fins a quina edat?»*), el cercador hi afegeix la pregunta anterior per saber de què es parla.
- **Model:** `gpt-4o-mini` es retira de l'API d'OpenAI el 14/1/2027; per això fem servir `gpt-6-luna`. Si canvia, només cal canviar `OPENAI_MODEL` als secrets.
- **Vies de millora:** registrar les preguntes **sense dades personals** per revisar-ne una mostra cada mes; filtrar per data de vigència; afegir els ajuts que falten.
