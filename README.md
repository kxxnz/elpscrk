# elpscrk — Intelligent Wordlist Generator (defensivo)

Gerador inteligente de wordlists baseado em profiling, permutações e estatísticas. Ferramenta inspirada no conceito visto em *Mr. Robot* — **uso exclusivamente defensivo/educacional**.

---

## Requisitos rápidos

- Python 3.8+ (acessível como `python`).
- (Opcional) criar e ativar um `venv` para isolar dependências.

### Dependências de desenvolvimento (para rodar testes)

Instale dependências de desenvolvimento com o `pip` quando fornecido o `requirements-dev.txt`:

```powershell
python -m pip install -r requirements-dev.txt
```

---

## Arquivos importantes

- **`elpscrk.py`** — script principal.
- **`banner.txt`** — banner exibido no início (opcionalmente colorido).
- **`README.md`** — este arquivo (instruções e exemplos).
- **`wordlist.txt`** — arquivo de saída (gerado pelo script).
- **`tests/`** — testes unitários.

---

## Campos do perfil (JSON)

Coloque um arquivo JSON com campos opcionais usados pelo gerador (ex.: `perfil.json`) e passe via `--profile perfil.json`.

Exemplo mínimo (`perfil.json`):

```json
{
  "names": ["Joao", "Pedro", "Cavalheiro", "dos", "Reis"],
  "nicknames": ["joaopedro", "jpreis", "joaowrlld", "kxxnz", "joaowrlld"],
  "usernames": ["joaowrlld", "kxxnz"],
  "company": ["Sicredi"],
  "pets": ["Faruk"],
  "hobbies": ["Futebol", "Python"],
  "places": ["Medianeira"],
  "dates": ["2007-05-22"],
  "numbers": ["42"],
  "corpus_text": "Joao loves futebol and python. Joao Pedro is often online."
}
```

> Use **apenas** dados aos quais você tem permissão para usar (testes/conta própria/laboratório).

---

## Formato aceito para arquivo de *breach* (lista pública de senhas)

- Uma senha por linha.
- Duas colunas (ex.: `count password` ou `password count`).
- Aceita arquivos comprimidos `.gz`.

Passe o arquivo com `--breach top100k.txt` ou `--breach top100k.txt.gz`.

---

## Comandos de uso (PowerShell)

Exibir ajuda:

```powershell
python elpscrk.py --help
```

Execução conservadora (modo `targeted`, padrão):

```powershell
python elpscrk.py
# gera e escreve `wordlist.txt` e mostra amostra na saída
```

Usar perfil e arquivo de breach:

```powershell
python elpscrk.py --profile perfil.json --breach top100k.txt
```

Ativar amostragem probabilística por Markov (mais variedade — bom em `broad`):

```powershell
python elpscrk.py --profile perfil.json --markov-sample --markov-temp 0.8
```

Modo amplo (`broad`) — cuidado, pode gerar muitos candidatos (respeite `--max`):

```powershell
python elpscrk.py --mode broad --max 500000
```

Suprimir banner:

```powershell
python elpscrk.py --no-banner
```

Escrever em arquivo customizado:

```powershell
python elpscrk.py --out meu_wordlist.txt
```

---

## Flags úteis (resumo)

- `--mode` : `targeted` (padrão) ou `broad`.
- `--profile` : caminho para JSON de perfil.
- `--breach` : arquivo de breaches (aceita `.gz`).
- `--markov-sample` : ativa amostragem Markov (probabilística).
- `--markov-temp` : temperatura da amostragem Markov (ex.: `0.7`, `1.0`).
- `--min-entropy` : filtrar candidatos por entropia (bits). Ex.: `14.0`, `18.0`.
- `--max` : cap de candidatos totais (evita explosão combinatória).
- `--no-banner` : não mostrar banner.
- `--no-color` : desativa cores no banner (se implementado).
- `--templates` : templates customizados, ex.: `"{name}{YY}"`. Pode passar vários.
- `--out` : arquivo de saída (default `wordlist.txt`).

---

## Entropia / Filtragem

`--min-entropy` utiliza um estimador simplificado (estilo NIST-ish). Se muitos candidatos forem filtrados, reduza esse valor (ex.: `14.0`) para gerar mais variações.

---

## Executar testes

Se quiser confirmar integração e qualidade do código:

```powershell
python -m pytest -q
```

(Os testes do projeto já passaram localmente.)

---

## Visualizar resultado

O arquivo `wordlist.txt` (ou o arquivo passado em `--out`) conterá as senhas geradas, ordenadas por pontuação. Para inspecionar as primeiras linhas no PowerShell:

```powershell
Get-Content wordlist.txt -TotalCount 50
```

---

## Segurança e uso responsável

Este script é destinado a auditoria defensiva. **Use somente** em sistemas/contas onde você tem autorização explícita. Atividades sem permissão são ilegais e eticamente erradas.

---

*Feito por @joaowrlld — mantenha o uso responsável.*

