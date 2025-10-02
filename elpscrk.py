import argparse
import itertools
import math
import json
import re
import sys
import random
from collections import Counter, defaultdict
from datetime import datetime
import gzip

# -----------------------------
# Helper utilities
# -----------------------------
def normalize_token(tok):
    tok = tok.strip()
    tok = re.sub(r'\s+', ' ', tok)
    return tok

def all_case_variants(s):
    """Return conservative case variants (lower, title, upper, capitalize first only)."""
    yield s.lower()
    yield s.title()
    yield s.upper()
    if len(s) > 1:
        yield s[0].upper() + s[1:].lower()

def leet_variants(s):
    """Substituições simples em leetspeak - limitadas para evitar explosão combinatória."""
    subs = {
        'a': ['4', '@'],
        'e': ['3'],
        'i': ['1', '!'],
        'o': ['0'],
        's': ['5', '$'],
        't': ['7'],
        'l': ['1']
    }
    # produce a few variations by substituting up to 2 chars
    variants = set([s])
    chars = list(s.lower())
    n = len(chars)
    for i in range(n):
        c = chars[i]
        if c in subs:
            for rep in subs[c]:
                new = chars[:]
                new[i] = rep
                variants.add(''.join(new))
    # also try two substitutions for short words
    for i in range(n):
        for j in range(i+1, n):
            ci, cj = chars[i], chars[j]
            if ci in subs and cj in subs:
                for ri in subs[ci]:
                    for rj in subs[cj]:
                        new = chars[:]
                        new[i] = ri
                        new[j] = rj
                        variants.add(''.join(new))
    return variants

def add_numeric_suffixes(s):
    """Add some common numeric suffixes (years, small numbers)."""
    now_year = datetime.utcnow().year
    out = set([s, s+'1', s+'123', s+'!'])
    for y in range(now_year-5, now_year+1):
        out.add(s + str(y))
    # common years (birth year patterns)
    for y in range(1970, 2026, 1):
        # include only if it matches small heuristics (avoid explosion)
        if s and len(s) <= 12 and (y % 10 == 0 or y % 5 == 0 or y % 100 == 0):
            out.add(s + str(y))
    return out

def estimate_entropy(s):
    """Estimativa simples de entropia estilo Shannon por caractere (aprox.)."""
    if not s:
        return 0.0
    counts = Counter(s)
    length = len(s)
    ent = -sum((c/length) * math.log2(c/length) for c in counts.values())
    # scale by length to get bits
    return ent * length


def nist_entropy_estimate(password):
    """A conservative, simplified NIST-like entropy estimator.
    This is not a full implementation, but gives a priors-aware score.
    """
    if not password:
        return 0.0
    # base: character set size heuristic
    lower = any(c.islower() for c in password)
    upper = any(c.isupper() for c in password)
    digits = any(c.isdigit() for c in password)
    symbols = any(not c.isalnum() for c in password)
    pool = 26 * (1 if lower else 0) + 26 * (1 if upper else 0) + 10 * (1 if digits else 0) + 32 * (1 if symbols else 0)
    pool = max(pool, 2)
    bits = math.log2(pool) * len(password)
    # small bonus for mixed classes
    classes = sum([lower, upper, digits, symbols])
    bits += (classes - 1) * 1.5
    return bits

# -----------------------------
# Core generator
# -----------------------------
class WordlistGenerator:
    def __init__(self, profile, max_candidates=100000, min_entropy=18.0):
        """
        profile: dict with keys like names, nicknames, words, dates, numbers
        max_candidates: safety cap to avoid explosion
        min_entropy: filter candidates below this estimated entropy
        """
        self.profile = profile
        self.max_candidates = max_candidates
        self.min_entropy = min_entropy
        self.tokens = self.extract_tokens()
        self.candidates = dict()  # token -> score
        # additional components (created lazily)
        self.markov = None
        self.breach = None
        self.templates = None
        self.kb = None

    def extract_tokens(self):
        t = set()
        # collect base tokens
        for k in ['names', 'nicknames', 'usernames', 'words', 'company', 'pets', 'hobbies', 'places']:
            for v in self.profile.get(k, []):
                v = normalize_token(str(v))
                if v:
                    # split multiword into parts too
                    parts = re.split(r'[\s._-]+', v)
                    for p in parts:
                        if p:
                            t.add(p)
                    t.add(v.replace(' ', ''))  # no-space version
        # dates and numbers as tokens
        for d in self.profile.get('dates', []):
            s = str(d)
            # accept YYYY, YY, DDMMYYYY combinations
            m = re.findall(r'\d+', s)
            for part in m:
                if len(part) >= 2:
                    t.add(part)
                    if len(part) == 4:
                        t.add(part[2:])  # yy
        for n in self.profile.get('numbers', []):
            t.add(str(n))
        # fallback: add common words from profile text corpus (if provided)
        corpus = self.profile.get('corpus_text', '')
        if corpus:
            words = re.findall(r"[A-Za-z0-9']{2,}", corpus)
            c = Counter([w.lower() for w in words])
            for w, freq in c.most_common(50):
                t.add(w)
        # filter trivial tokens
        tokens = set(x for x in t if len(x) >= 2 and not re.fullmatch(r'\W+', x))
        return tokens

    def attach_markov(self, markov_model):
        self.markov = markov_model

    def attach_breach(self, breach_obj):
        self.breach = breach_obj

    def attach_template_expander(self, expander):
        self.templates = expander

    def attach_keyboard(self, kb):
        self.kb = kb

    def score_candidate(self, cand):
        """Score combines frequency heuristics and entropy. Higher is better."""
        score = 0.0
        # small boost for containing profile tokens
        for tok in self.tokens:
            if tok in cand.lower():
                score += 1.0
        # penalize extremely short or extremely long
        if 6 <= len(cand) <= 20:
            score += 1.0
        # entropy boost
        e = estimate_entropy(cand)
        nist = nist_entropy_estimate(cand)
        score += (e / 8.0)  # scale down
        score += (nist / 20.0)
        # breach frequency penalty (if candidate appears in breaches, lower score but still keep)
        if self.breach:
            freq = self.breach.get_freq(cand)
            if freq is not None:
                # higher frequency -> more likely, give a small boost for prioritization during defensive analysis
                score += math.log1p(1 + freq) * 0.5
        # markov plausibility (if available)
        if self.markov:
            score += self.markov.score_sequence(cand)
        return score

    def generate(self):
        # Conservative generation pipeline combining templates, token combos, markov and mutators
        produced = set()
        # 1) Expand templates if available
        candidates_src = []
        if self.templates:
            candidates_src.extend(self.templates.expand(self.tokens))
        # 2) single tokens and pairwise (limited)
        token_list = list(self.tokens)
        for t in token_list:
            candidates_src.append(t)
        for a, b in itertools.permutations(token_list, 2):
            if a == b:
                continue
            candidates_src.append(a + b)
            if len(candidates_src) > self.max_candidates // 3:
                break
        # 3) markov-generated candidates (conservative)
        if self.markov:
            for m in self.markov.generate(n=200, max_len=12):
                candidates_src.append(m)

        # apply transforms conservatively and score
        for base in candidates_src:
            base = str(base)
            # case and leet variants
            variants = set()
            for cv in all_case_variants(base):
                variants.add(cv)
                for lv in leet_variants(cv):
                    variants.add(lv)
            # numeric suffixes
            to_add = set()
            for v in variants:
                to_add.update(add_numeric_suffixes(v))
            # keyboard mutations if available (limit depth)
            if self.kb:
                mutated = set()
                for v in list(to_add)[:50]:
                    mutated.update(self.kb.mutate(v, depth=1, max_variants=8))
                to_add.update(mutated)

            # filter and score
            for cand in to_add:
                if cand in produced:
                    continue
                produced.add(cand)
                if nist_entropy_estimate(cand) < self.min_entropy:
                    continue
                score = self.score_candidate(cand)
                self.candidates[cand] = score
            if len(self.candidates) >= self.max_candidates:
                break
        # After generating, return a ranked list
        ranked = sorted(self.candidates.items(), key=lambda x: x[1], reverse=True)
        return ranked


class MarkovModel:
    """Modelo de Markov em nível de caractere (ordem configurável). Amostragem conservadora por padrão."""
    def __init__(self, order=2):
        self.order = order
        self.transitions = defaultdict(Counter)
        self.starts = Counter()

    def set_sampling(self, enabled=False, temperature=1.0):
        self.sampling = enabled
        self.temperature = max(0.1, float(temperature))

    def train(self, corpus):
        for line in corpus.splitlines():
            s = line.strip()
            if not s:
                continue
            padded = ('^' * self.order) + s + '$'
            self.starts[padded[:self.order]] += 1
            for i in range(len(padded)-self.order):
                prev = padded[i:i+self.order]
                nxt = padded[i+self.order]
                self.transitions[prev][nxt] += 1

    def score_sequence(self, seq):
        # return a small log-probability-like score (conservative)
        if not seq:
            return 0.0
        padded = ('^' * self.order) + seq + '$'
        score = 0.0
        for i in range(len(padded)-self.order):
            prev = padded[i:i+self.order]
            nxt = padded[i+self.order]
            counts = self.transitions.get(prev)
            if not counts:
                score -= 1.0
            else:
                total = sum(counts.values())
                score += math.log1p(counts.get(nxt, 0) + 0.1) - math.log1p(total + 0.1)
        return score

    def generate(self, n=100, max_len=12):
        out = []
        starts = list(self.starts.items())
        if not starts:
            return out
        for _ in range(n):
            # conservative sample: choose frequent start
            start = max(starts, key=lambda x: x[1])[0]
            cur = start
            s = ''
            for _ in range(max_len):
                choices = self.transitions.get(cur)
                if not choices:
                    break
                # probabilistic sampling if enabled, otherwise most-likely
                if getattr(self, 'sampling', False):
                    items = list(choices.items())
                    chars, weights = zip(*items)
                    # apply temperature
                    w = [pow(float(x), 1.0/self.temperature) for x in weights]
                    total = sum(w)
                    probs = [x/total for x in w]
                    nxt = random.choices(chars, probs, k=1)[0]
                else:
                    nxt = max(choices.items(), key=lambda x: x[1])[0]
                if nxt == '$':
                    break
                s += nxt
                cur = (cur + nxt)[-self.order:]
            if s:
                out.append(s)
        return list(dict.fromkeys(out))


class TemplateExpander:
    def __init__(self, user_templates=None):
        # built-in templates
        self.builtin = ["{name}{YYYY}", "{pet}{!!}", "{word}{123}", "{name|nick}{YY}"]
        self.user_templates = user_templates or []

    def expand(self, tokens):
        tokens = list(tokens)
        results = set()
        templates = self.builtin + self.user_templates
        # prepare date parts
        now = datetime.utcnow()
        years = [str(y) for y in range(now.year-40, now.year+1)]
        months = [f"{m:02d}" for m in range(1,13)]
        days = [f"{d:02d}" for d in range(1,32)]
        for tpl in templates:
            # support token-sets like {name|nick} and placeholders {YYYY},{YY},{MM},{DD},{MMDD},{!!},{123}
            for t in tokens:
                # build a small context map for replacements
                ctx = {
                    'name': t,
                    'word': t,
                    'pet': t,
                    'nick': t,
                }
                # handle token-set patterns: {a|b}
                # We'll expand token sets by replacing with available tokens when matching the names
                # naive approach: if tpl contains '|', attempt to substitute any token label
                raw = tpl
                # replace simple placeholders first
                for key, val in ctx.items():
                    raw = raw.replace('{' + key + '}', val)
                raw = raw.replace('{!!}', '!!').replace('{123}', '123')
                # expand years/months/days
                for y in years[:10]:
                    r1 = raw.replace('{YYYY}', y).replace('{YY}', y[-2:])
                    for mm in months:
                        r2 = r1.replace('{MM}', mm)
                        for dd in days[:3]:
                            # limit days expansion to first few to avoid explosion
                            r3 = r2.replace('{DD}', dd).replace('{MMDD}', mm+dd)
                            results.add(r3)
        # also handle user-provided templates that may use token-sets like {name|nick}
        # naive additional pass: if a template contains '|' try simple splits
        for tpl in self.user_templates:
            if '|' in tpl:
                parts = re.findall(r'\{([^}]+)\}', tpl)
                # for each token in tokens, attempt replacements
                for t in tokens:
                    s = tpl
                    # for each bracketed part containing '|', choose token if known
                    for part in parts:
                        if '|' in part:
                            # prefer name/nick mapping
                            if 'name' in part:
                                s = s.replace('{' + part + '}', t)
                            else:
                                s = s.replace('{' + part + '}', t)
                    results.add(s)
        return results


class KeyboardMutator:
    def __init__(self, layout='qwerty'):
        # approximate neighbor map for QWERTY (lowercase)
        self.neighbors = {
            'a': 'qwsz', 'b': 'vghn', 'c': 'xdfv', 'd': 'ersfcx', 'e': 'wsdr',
            'f': 'rtgdvc', 'g': 'tyfhvb', 'h': 'yugjbn', 'i': 'ujko', 'j': 'uikhmn',
            'k': 'jiolm', 'l': 'kop', 'm': 'njk', 'n': 'bhjm', 'o': 'iklp',
            'p': 'ol', 'q': 'wa', 'r': 'etdf', 's': 'awedxz', 't': 'ryfg',
            'u': 'yhji', 'v': 'cfgb', 'w': 'qase', 'x': 'zsdc', 'y': 'tghu',
            'z': 'asx', '1': '2q', '2': '13w', '3': '24e', '4': '35r', '5': '46t',
            '6': '57y', '7': '68u', '8': '79i', '9': '80o', '0': '9p'
        }

    def mutate(self, s, depth=1, max_variants=10):
        s = str(s)
        variants = set([s])
        for _ in range(depth):
            new = set()
            for v in variants:
                for i, ch in enumerate(v):
                    lower = ch.lower()
                    if lower in self.neighbors:
                        for nb in self.neighbors[lower][:3]:
                            cand = v[:i] + nb + v[i+1:]
                            new.add(cand)
            variants.update(new)
            if len(variants) > max_variants:
                break
        # keep casing variants limited
        out = set()
        for v in list(variants)[:max_variants]:
            out.update(all_case_variants(v))
        return out


class BreachList:
    def __init__(self, path=None):
        self.freq = {}
        if path:
            try:
                opener = open
                if path.endswith('.gz'):
                    opener = gzip.open
                with opener(path, 'rt', encoding='utf-8', errors='ignore') as f:
                    for i, line in enumerate(f):
                        line = line.strip()
                        if not line:
                            continue
                        # common formats: 'password' or 'count password' or 'password count'
                        parts = line.split()
                        if len(parts) == 1:
                            pw = parts[0]
                            cnt = max(1, 100000 - i)
                        elif len(parts) == 2:
                            # guess which is numeric
                            if parts[0].isdigit():
                                cnt = int(parts[0])
                                pw = parts[1]
                            elif parts[1].isdigit():
                                cnt = int(parts[1])
                                pw = parts[0]
                            else:
                                pw = parts[0]
                                cnt = max(1, 100000 - i)
                        else:
                            pw = parts[0]
                            cnt = max(1, 100000 - i)
                        self.freq[pw] = cnt
            except Exception:
                # failed to load; keep empty
                pass

    def get_freq(self, token):
        return self.freq.get(token)


def load_profile_from_file(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Falha ao carregar perfil de {path}: {e}")
        return {}


def load_banner(path='banner.txt'):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception:
        return None


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='Gerador inteligente de wordlists (modo conservador/targeted)')
    p.add_argument('--profile', help='Caminho para JSON de perfil', default=None)
    p.add_argument('--mode', choices=['targeted', 'broad'], default='targeted', help='Modo: targeted (focado) ou broad (amplo)')
    p.add_argument('--out', '-o', help='Arquivo de saída', default='wordlist.txt')
    p.add_argument('--max', type=int, default=20000, help='Máximo de candidatos (cap)')
    p.add_argument('--min-entropy', type=float, default=18.0, help='Entropia mínima (bits) para filtrar candidatos')
    p.add_argument('--breach', help='Arquivo de breaches (opcional, suporta .gz e formatos comuns)')
    p.add_argument('--templates', nargs='*', help='Templates do usuário (ex: "{name}{YY}")')
    p.add_argument('--keyboard-depth', type=int, default=1, help='Profundidade de mutação de teclado')
    p.add_argument('--markov-sample', action='store_true', help='Ativa amostragem Markov (probabilística)')
    p.add_argument('--markov-temp', type=float, default=1.0, help='Temperatura para amostragem Markov (ex: 0.8)')
    p.add_argument('--no-banner', action='store_true', help='Não exibir banner ASCII no início')
    p.add_argument('--show-sample', action='store_true', help='Mostrar amostra dos candidatos gerados no terminal')
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    banner = load_banner()
    if banner and not args.no_banner:
        print(banner)
    profile = {}
    if args.profile:
        profile = load_profile_from_file(args.profile)
    else:
        # perfil padrão conservador (exemplo)
        profile = {
            'names': ['Joao', 'Pedro'],
            'nicknames': ['peter'],
            'usernames': ['joaowrlld'],
            'company': ['Sicredi'],
            'pets': ['Faruk'],
            'hobbies': ['futebol', 'python'],
            'places': ['Medianeira'],
            'dates': ['2007-05-22'],
            'numbers': ['42'],
            'corpus_text': "Joao loves futebol and python. JoaoPedro is often online."
        }

    gen = WordlistGenerator(profile, max_candidates=args.max, min_entropy=args.min_entropy)

    # anexar componentes
    markov = MarkovModel(order=2)
    # train on corpus and tokens conservatively
    corpus = profile.get('corpus_text', '') + '\n' + '\n'.join(profile.get('usernames', []))
    markov.train(corpus)
    # configurar amostragem Markov (se solicitado)
    markov.set_sampling(enabled=args.markov_sample, temperature=args.markov_temp)
    gen.attach_markov(markov)

    breach = BreachList(path=args.breach) if args.breach else None
    gen.attach_breach(breach)

    tpl = TemplateExpander(user_templates=args.templates)
    gen.attach_template_expander(tpl)

    kb = KeyboardMutator()
    gen.attach_keyboard(kb)

    # limites baseados em modo
    if args.mode == 'targeted':
        gen.max_candidates = min(gen.max_candidates, 5000)
    else:
        # modo broad - aviso sobre tamanho
        print('Modo broad: cuidado, isto pode gerar muitos candidatos. Use --max para limitar o tamanho.')

    ranked = gen.generate()

    # output
    out_n = 1000 if args.mode == 'targeted' else min(len(ranked), gen.max_candidates)
    with open(args.out, 'w', encoding='utf-8') as f:
        for pw, score in ranked[:out_n]:
            f.write(pw + '\n')

    if args.show_sample:
        print(f"Escreveu {min(out_n,len(ranked))} candidatos em {args.out}")
        for pw, score in ranked[:50]:
            print(f"{pw:25s}  pontuação={score:.2f}  entropia={estimate_entropy(pw):.2f}")


if __name__ == '__main__':
    main()
