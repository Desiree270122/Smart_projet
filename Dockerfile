# Application web 2SMART dans un seul conteneur : l'interface (web/) est
# construite, puis servie par le serveur de calcul (api/).
#
#   docker build -t 2smart .
#   docker run -p 8000:8000 2smart        puis ouvrir http://localhost:8000
#
# Le port d'écoute suit la variable PORT quand l'hébergeur en impose un.

FROM node:22-slim AS interface
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt requirements-web.txt ./
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-web.txt
COPY . .
COPY --from=interface /web/dist ./web/dist
ENV PORT=8000
EXPOSE 8000
# Un seul processus : les simulations lancées sont gardées en mémoire.
CMD ["sh", "-c", "python -m uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
