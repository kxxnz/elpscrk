import itertools
import math
import json
import re
from collections import Counter, defaultdict
from datetime import datetime

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
    """Simple leetspeak substitutions - keep limited to avoid explosion."""
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
    """Simple Shannon-style entropy estimate per-character (approx)."""
    if not s:
        return 0.0
    counts = Counter(s)
    length = len(s)
    ent = -sum((c/length) * math.log2(c/length) for c in counts.values())
    # scale by length to get bits
    return ent * length

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
        score += (e / 8.0)  # scale down
        return score

    def generate(self):
        # Basic combinatorics: combine 1..2 tokens (targeted)
        token_list = list(self.tokens)
        combos = []
        # prioritize single tokens and pairwise
        for t in token_list:
            combos.append((t,))
        for a, b in itertools.permutations(token_list, 2):
            # skip identical repeats to reduce explosion
            if a == b:
                continue
            combos.append((a,b))
            if len(combos) > self.max_candidates // 4:
                break
        # transforms and ranking
        produced = set()
        for combo in combos:
            base = ''.join(combo)
            variants = set()
            # case variants
            for cv in all_case_variants(base):
                variants.add(cv)
                # leet variants (a few)
                for lv in leet_variants(cv):
                    variants.add(lv)
            # add numerics and common suffixes
            to_add = set()
            for v in variants:
                to_add.update(add_numeric_suffixes(v))
            # also insert common separators for readability (rarely used in passwords)
            for v in list(to_add):
                to_add.add(v + '_' )
                to_add.add(v + '-' )
                to_add.add('!' + v)
            # evaluate and keep
            for cand in to_add:
                if cand in produced:
                    continue
                produced.add(cand)
                # filter by entropy
                if estimate_entropy(cand) < self.min_entropy:
                    continue
                score = self.score_candidate(cand)
                self.candidates[cand] = score
            if len(self.candidates) >= self.max_candidates:
                break
        # After generating, return a ranked list
        ranked = sorted(self.candidates.items(), key=lambda x: x[1], reverse=True)
        return ranked

# -----------------------------
# Example usage (fill profile)
# -----------------------------
if __name__ == '__main__':
    profile = {
        'names': ['Joao', 'Pedro', 'Cavalheiro', 'dos', 'Reis'],
        'nicknames': ['peter', 'jpreis'],
        'usernames': ['joaowrlld'],
        'company': ['Sicredi'],
        'pets': ['Faruk'],
        'hobbies': ['futebol', 'python'],
        'places': ['Medianeira'],
        'dates': ['2007-05-22'],
        'numbers': ['42'],
        'corpus_text': "Joao loves futebol and python. JoaoPedro is often online."
    }

    gen = WordlistGenerator(profile, max_candidates=20000, min_entropy=18.0)
    ranked = gen.generate()
    # write top N to file
    N = 2000
    out_file = 'wordlist.txt'
    with open(out_file, 'w', encoding='utf-8') as f:
        for pw, score in ranked[:N]:
            f.write(pw + '\n')
    print(f"Wrote top {min(N,len(ranked))} candidates to {out_file}")
    # show some sample ranked entries
    for pw, score in ranked[:30]:
        print(f"{pw:25s}  score={score:.2f}  entropy={estimate_entropy(pw):.2f}")
