# -*- coding: utf-8 -*-
"""
Semantic Banking Guardrail Module (Layer 2 - Defense-in-Depth)
=============================================================
Detects sophisticated adversarial attacks, bypass attempts, jailbreaks, and
banking fraud based on semantic cosine similarity against an Attack Vector Bank
and a Benign Banking Template Bank using Contrastive Guardrail verification.

DESIGN INVARIANTS:
1. 0% REGEX LAW: No regex methods or import re.
2. ZERO TERNARY LAW: Explicit if/else statements only.
3. NO TYPING LIBRARY: No typing library imports.
4. NO __INIT__.PY: Direct module imports.
"""

from agent_banking.core.settings import Config
from agent_banking.guardrails.config.settings import GuardrailsConfig, load_guardrail_prompts

try:
    import numpy as np
    HAVE_NUMPY = True
except Exception:
    np = None
    HAVE_NUMPY = False

# Global process-lifetime in-memory vector cache
_GLOBAL_ATTACK_VECTORS = []
_GLOBAL_BENIGN_VECTORS = []
_GLOBAL_ATTACK_MATRIX = None
_GLOBAL_BENIGN_MATRIX = None
_GLOBAL_INITIALIZED = False
_GLOBAL_EMBEDDER = None

_prompts = load_guardrail_prompts()
_semantic_cfg = dict(GuardrailsConfig.SEMANTIC_SETTINGS or {})
_semantic_logs = dict(getattr(_prompts, "SEMANTIC_LOG_MESSAGES", {}) or {})

SEMANTIC_ATTACK_TEMPLATES = dict(getattr(_prompts, "SEMANTIC_ATTACK_TEMPLATES", {}) or {})
BENIGN_BANKING_TEMPLATES = list(getattr(_prompts, "BENIGN_BANKING_TEMPLATES", []) or [])

_MIN_QUERY_LENGTH = int(_semantic_cfg.get("min_query_length", 0))
_BENIGN_MIN_SCORE = float(_semantic_cfg.get("benign_min_score", 0.0))
_BENIGN_ATTACK_MARGIN = float(_semantic_cfg.get("benign_attack_margin", 0.0))
_ROUND_DIGITS = int(_semantic_cfg.get("similarity_round_digits", 0))
_DEFAULT_EMBEDDING_DIM = int(_semantic_cfg.get("default_embedding_dim", 0))
_QUERY_PREFIX = str(_semantic_cfg.get("query_prefix") or "")


def _log_msg(key, **kwargs):
    template = str(_semantic_logs.get(key) or "")
    if not template:
        return ""
    return template.format(**kwargs)


class SemanticBankingGuardrail:
    """
    Semantic banking guardrail using dense embedding vectors with contrastive verification.
    """

    def __init__(self, embedder=None, threshold=None, decision_client=None, use_decision_model=True):
        self.embedder = embedder
        if threshold is None:
            self.threshold = GuardrailsConfig.SEMANTIC_SIMILARITY_THRESHOLD
        else:
            self.threshold = threshold
        self.use_decision_model = use_decision_model
        if decision_client is not None:
            self.decision_client = decision_client
        else:
            self.decision_client = None
        self.attack_bank_vectors = []
        self.benign_bank_vectors = []
        self._attack_matrix = None
        self._benign_matrix = None
        self._is_initialized = False

    def _ensure_initialized(self):
        """Initialize and compute vectors for Attack and Benign template banks (lazy load)."""
        global _GLOBAL_INITIALIZED, _GLOBAL_ATTACK_VECTORS, _GLOBAL_BENIGN_VECTORS
        global _GLOBAL_ATTACK_MATRIX, _GLOBAL_BENIGN_MATRIX, _GLOBAL_EMBEDDER

        if self.embedder is None:
            if _GLOBAL_EMBEDDER is not None:
                self.embedder = _GLOBAL_EMBEDDER
            else:
                from agent_banking.embedding.embedder import BankingEmbedder
                self.embedder = BankingEmbedder(
                    dimension=getattr(Config, "EMBEDDING_DIM", _DEFAULT_EMBEDDING_DIM),
                    model_name=getattr(Config, "EMBEDDING_MODEL_1024", getattr(Config, "EMBEDDING_MODEL", "")),
                    api_key=Config.AGENT_LLM_API_KEY,
                    embedding_url=getattr(Config, "EMBEDDING_URL", "")
                )
                _GLOBAL_EMBEDDER = self.embedder

        if self._is_initialized:
            return

        if _GLOBAL_INITIALIZED:
            self.attack_bank_vectors = _GLOBAL_ATTACK_VECTORS
            self.benign_bank_vectors = _GLOBAL_BENIGN_VECTORS
            self._attack_matrix = _GLOBAL_ATTACK_MATRIX
            self._benign_matrix = _GLOBAL_BENIGN_MATRIX
            self._is_initialized = True
            return

        attack_items = []
        for category, templates in SEMANTIC_ATTACK_TEMPLATES.items():
            for tpl in templates:
                attack_items.append({"category": category, "template": tpl})

        benign_items = []
        for b_tpl in BENIGN_BANKING_TEMPLATES:
            benign_items.append({"template": b_tpl})

        use_query_prefix = False
        if hasattr(self.embedder, "use_query_prefix"):
            use_query_prefix = bool(self.embedder.use_query_prefix)
        elif hasattr(self.embedder, "prefix_query"):
            use_query_prefix = bool(self.embedder.prefix_query)

        hashes = []
        for item in attack_items:
            tpl = item["template"]
            if use_query_prefix:
                vp = _QUERY_PREFIX + tpl
            else:
                vp = tpl
            if hasattr(self.embedder, "_get_hash"):
                h = self.embedder._get_hash(vp)
            else:
                import hashlib
                h = hashlib.sha256(vp.encode("utf-8")).hexdigest()
            hashes.append(h)

        for b_item in benign_items:
            b_tpl = b_item["template"]
            if use_query_prefix:
                b_vp = _QUERY_PREFIX + b_tpl
            else:
                b_vp = b_tpl
            if hasattr(self.embedder, "_get_hash"):
                b_h = self.embedder._get_hash(b_vp)
            else:
                import hashlib
                b_h = hashlib.sha256(b_vp.encode("utf-8")).hexdigest()
            hashes.append(b_h)

        cached_vectors = {}
        if hasattr(self.embedder, "_get_cached_vectors_batch"):
            try:
                cached_vectors = self.embedder._get_cached_vectors_batch(hashes)
            except Exception as e:
                print(_log_msg("batch_cache_query_error", error=e))
                cached_vectors = {}

        self.attack_bank_vectors = []
        for idx, item in enumerate(attack_items):
            h = hashes[idx]
            vec = None
            if h in cached_vectors:
                vec = cached_vectors[h]
            else:
                try:
                    vec = self.embedder.embed_query(item["template"])
                except Exception as e:
                    print(_log_msg("embed_attack_error", error=e))
            if vec is not None:
                self.attack_bank_vectors.append({
                    "category": item["category"],
                    "template": item["template"],
                    "vector": vec
                })

        self.benign_bank_vectors = []
        offset = len(attack_items)
        for idx, b_item in enumerate(benign_items):
            h = hashes[offset + idx]
            b_vec = None
            if h in cached_vectors:
                b_vec = cached_vectors[h]
            else:
                try:
                    b_vec = self.embedder.embed_query(b_item["template"])
                except Exception as e:
                    print(_log_msg("embed_benign_error", error=e))
            if b_vec is not None:
                self.benign_bank_vectors.append({
                    "template": b_item["template"],
                    "vector": b_vec
                })

        if HAVE_NUMPY:
            if self.attack_bank_vectors:
                mat = np.array([x["vector"] for x in self.attack_bank_vectors], dtype=np.float32)
                norms = np.linalg.norm(mat, axis=1, keepdims=True)
                norms[norms == 0.0] = 1.0
                self._attack_matrix = mat / norms
            else:
                self._attack_matrix = None

            if self.benign_bank_vectors:
                b_mat = np.array([x["vector"] for x in self.benign_bank_vectors], dtype=np.float32)
                b_norms = np.linalg.norm(b_mat, axis=1, keepdims=True)
                b_norms[b_norms == 0.0] = 1.0
                self._benign_matrix = b_mat / b_norms
            else:
                self._benign_matrix = None
        else:
            self._attack_matrix = None
            self._benign_matrix = None

        _GLOBAL_ATTACK_VECTORS = self.attack_bank_vectors
        _GLOBAL_BENIGN_VECTORS = self.benign_bank_vectors
        _GLOBAL_ATTACK_MATRIX = self._attack_matrix
        _GLOBAL_BENIGN_MATRIX = self._benign_matrix
        _GLOBAL_INITIALIZED = True
        self._is_initialized = True

    def _cosine_similarity(self, vec_a, vec_b):
        """Compute Cosine Similarity between 2 L2-normalized vectors (dot product)."""
        if not vec_a:
            return 0.0
        if not vec_b:
            return 0.0
        if len(vec_a) != len(vec_b):
            return 0.0

        dot_product = 0.0
        for i in range(len(vec_a)):
            dot_product += vec_a[i] * vec_b[i]
        return dot_product

    def check_query(self, query, use_decision_model=None):
        """
        Check user query with contrastive semantic matching and optional decision engine.
        Returns:
            is_safe (bool): True if benign, False if attack suspected.
            category (str | None): Violation category.
            max_similarity (float): Highest similarity score.
            matched_template (str | None): Closest template.
        """
        if not query:
            return {
                "is_safe": True,
                "category": None,
                "max_similarity": 0.0,
                "matched_template": None
            }

        clean_query = query.strip()
        if len(clean_query) < _MIN_QUERY_LENGTH:
            return {
                "is_safe": True,
                "category": None,
                "max_similarity": 0.0,
                "matched_template": None
            }

        should_use_decision = self.use_decision_model
        if use_decision_model is not None:
            should_use_decision = use_decision_model

        if should_use_decision is True:
            if self.decision_client is not None:
                try:
                    if self.decision_client.is_available():
                        audit = self.decision_client.audit_adversarial_input(clean_query)
                        if isinstance(audit, dict):
                            is_adv = False
                            if audit.get("is_adversarial") is True:
                                is_adv = True
                            elif audit.get("is_attack") is True:
                                is_adv = True
                            elif audit.get("choice") == "ATTACK":
                                is_adv = True

                            if is_adv is True:
                                return {
                                    "is_safe": False,
                                    "category": "PROMPT_INJECTION_DETECTED",
                                    "max_similarity": 1.0,
                                    "matched_template": str(_semantic_logs.get("decision_audit_template_label") or ""),
                                    "decision_audit": audit
                                }
                except Exception as e:
                    print(_log_msg("decision_client_fallback", error=e))

        try:
            self._ensure_initialized()
        except Exception as e:
            print(_log_msg("init_failed", error=e))
            return {
                "is_safe": True,
                "category": None,
                "max_similarity": 0.0,
                "matched_template": None
            }

        if not self.attack_bank_vectors:
            return {
                "is_safe": True,
                "category": None,
                "max_similarity": 0.0,
                "matched_template": None
            }

        try:
            query_vec = self.embedder.embed_query(clean_query)
        except Exception as e:
            print(_log_msg("query_embed_error", error=e))
            return {
                "is_safe": True,
                "category": None,
                "max_similarity": 0.0,
                "matched_template": None
            }

        best_benign_score = 0.0
        best_attack_score = 0.0
        best_category = None
        best_template = None

        use_matrix = False
        if HAVE_NUMPY:
            if self._attack_matrix is not None:
                if self._benign_matrix is not None:
                    use_matrix = True

        if use_matrix:
            q_arr = np.array(query_vec, dtype=np.float32)
            q_norm = float(np.linalg.norm(q_arr))
            if q_norm > 0.0:
                q_arr = q_arr / q_norm

            if len(self.benign_bank_vectors) > 0:
                b_scores = np.dot(self._benign_matrix, q_arr)
                best_benign_idx = int(np.argmax(b_scores))
                best_benign_score = float(b_scores[best_benign_idx])

            if len(self.attack_bank_vectors) > 0:
                a_scores = np.dot(self._attack_matrix, q_arr)
                best_attack_idx = int(np.argmax(a_scores))
                best_attack_score = float(a_scores[best_attack_idx])
                best_category = self.attack_bank_vectors[best_attack_idx]["category"]
                best_template = self.attack_bank_vectors[best_attack_idx]["template"]
        else:
            for b_item in self.benign_bank_vectors:
                score_b = self._cosine_similarity(query_vec, b_item["vector"])
                if score_b > best_benign_score:
                    best_benign_score = score_b

            for item in self.attack_bank_vectors:
                score = self._cosine_similarity(query_vec, item["vector"])
                if score > best_attack_score:
                    best_attack_score = score
                    best_category = item["category"]
                    best_template = item["template"]

        if best_benign_score >= _BENIGN_MIN_SCORE and best_benign_score >= (best_attack_score - _BENIGN_ATTACK_MARGIN):
            return {
                "is_safe": True,
                "category": None,
                "max_similarity": round(best_attack_score, _ROUND_DIGITS),
                "matched_template": best_template,
                "benign_similarity": round(best_benign_score, _ROUND_DIGITS)
            }

        if best_attack_score >= self.threshold:
            return {
                "is_safe": False,
                "category": best_category,
                "max_similarity": round(best_attack_score, _ROUND_DIGITS),
                "matched_template": best_template
            }

        return {
            "is_safe": True,
            "category": None,
            "max_similarity": round(best_attack_score, _ROUND_DIGITS),
            "matched_template": best_template
        }
