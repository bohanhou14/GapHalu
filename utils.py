from eyecite import get_citations, clean_text, resolve_citations
from eyecite.models import FullCaseCitation, CaseCitation

def extract_citations(text):
    # print(text)
    text = clean_text(text, ['html', 'all_whitespace'])
    citations = get_citations(text)
    citations = list(citations)
    resolutions = resolve_citations(citations)
    cases = [res for res in resolutions if (isinstance(res.citation, FullCaseCitation) or isinstance(res.citation, CaseCitation))]
    case_labels = [c.citation.token.data for c in cases]
    case_pos = [(c.citation.token.start, c.citation.token.end) for c in cases]
    return case_labels, case_pos