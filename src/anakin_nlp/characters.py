"""Character normalisation and faction labels.

Two deliberate departures from the original project:

1. **No fuzzy name matching.** Comparing every uppercase token against every other
   with ``SequenceMatcher`` at a 0.8 ratio does fix real typos (``ANKAIN``), but it
   also rewrites other words: in the original corrected files ``MACE`` had become
   ``MENACE`` and ``KNIGHT`` had become ``NIGHT``. The typos in these three scripts are
   few, so they are enumerated. Every alias below was read off the real cue frequencies.

2. **Faction labels are conservative.** The target is *not* "good vs evil", which is
   unfalsifiable, but membership of the two rhetorical camps: the Jedi/Republic
   side and the Sith/Separatist side. Mercenaries, criminals, bystanders and
   bureaucrats have no stable camp, so they are excluded and the exclusion is
   documented rather than silently resolved. A noisy label is worse than no label.
"""

from __future__ import annotations

SUBJECT = "ANAKIN"

#: Raw cue -> canonical character. Covers OCR/transcription typos and the several
#: names one character is credited under (personas, ranks, first-name-only cues).
ALIASES: dict[str, str] = {
    # --- the subject: Anakin and his Sith persona are one person ---------------
    "ANAKIN": "ANAKIN", "ANKAIN": "ANAKIN", "ANAKINN": "ANAKIN",
    "DARTH VADER": "ANAKIN", "VADER": "ANAKIN", "LORD VADER": "ANAKIN",
    # --- Padme: "Amidala" is the queen/senator persona of the same character ---
    "PADME": "PADME", "PADMÉ": "PADME", "PAMDÉ": "PADME",
    "AMIDALA": "PADME", "SENATOR AMIDALA": "PADME", "QUEEN AMIDALA": "PADME",
    # --- Palpatine: public Chancellor and secret Sith Lord are one person ------
    "PALPATINE": "PALPATINE", "PALAPATINE": "PALPATINE",
    "CHANCELLOR PALPATINE": "PALPATINE", "SUPREME CHANCELLOR": "PALPATINE",
    "DARTH SIDIOUS": "PALPATINE", "SIDIOUS": "PALPATINE", "DABTH SIDIOUS": "PALPATINE",
    # --- Jedi and Republic ----------------------------------------------------
    "QUI-GON": "QUI-GON", "QUI -GON": "QUI-GON", "GUI-GON": "QUI-GON",
    "QU-IG0N": "QUI-GON", "QUIGON": "QUI-GON",
    "OBI-WAN": "OBI-WAN", "OBIWAN": "OBI-WAN", "BEN": "OBI-WAN",
    "YODA": "YODA",
    "MACE": "MACE WINDU", "MACE WINDU": "MACE WINDU", "WINDU": "MACE WINDU",
    "KI-ADI": "KI-ADI-MUNDI", "KI-ADI-MUNDI": "KI-ADI-MUNDI", "KIADI": "KI-ADI-MUNDI",
    "PLO KOON": "PLO KOON", "AAYLA": "AAYLA", "JOCASTA NU": "JOCASTA NU",
    "SHMI": "SHMI", "JAR JAR": "JAR JAR", "JAR": "JAR JAR",
    "BAIL": "BAIL ORGANA", "BAIL ORGANA": "BAIL ORGANA", "ORGANA": "BAIL ORGANA",
    "MON MOTHMA": "MON MOTHMA", "MOTHMA": "MON MOTHMA",
    "CAPT. PANAKA": "PANAKA", "CAPTAIN PANAKA": "PANAKA", "PANAKA": "PANAKA",
    "CAPTAIN TYPHO": "TYPHO", "TYPHO": "TYPHO",
    "SIO BIBBLE": "SIO BIBBLE", "BIBBLE": "SIO BIBBLE",
    "C-3PO": "C-3PO", "THREEPIO": "C-3PO", "G-3PO": "C-3PO",
    "VALORUM": "VALORUM", "RIC OLIE": "RIC OLIE", "OLIE": "RIC OLIE",
    "BOSS NASS": "BOSS NASS", "NASS": "BOSS NASS",
    "CAPT. TARPALS": "TARPALS", "TARPALS": "TARPALS",
    "GENERAL CEEL": "GENERAL CEEL", "QUEEN JAMILLIA": "JAMILLIA", "JAMILLIA": "JAMILLIA",
    "DEXTER JETTSTER": "DEXTER JETTSTER", "JETTSTER": "DEXTER JETTSTER",
    "SOLA": "SOLA", "JOBAL": "JOBAL", "JOVAL": "JOBAL", "RUWEE": "RUWEE",
    "OWEN": "OWEN", "BERU": "BERU", "CLIEGG": "CLIEGG",
    "DORMÉ": "DORME", "DORME": "DORME", "RABE": "RABE", "SABE": "SABE",
    "KITSTER": "KITSTER", "CAPTAIN ANTILLES": "ANTILLES",
    # --- Sith, Separatists, Trade Federation ----------------------------------
    "DARTH MAUL": "DARTH MAUL", "MAUL": "DARTH MAUL",
    "COUNT DOOKU": "DOOKU", "DOOKU": "DOOKU",
    "GENERAL GRIEVOUS": "GRIEVOUS", "GRIEVOUS": "GRIEVOUS",
    "NUTE": "NUTE GUNRAY", "NUTE GUNRAY": "NUTE GUNRAY", "GUNRAY": "NUTE GUNRAY",
    "RUNE": "RUNE HAAKO", "RUNE HAAKO": "RUNE HAAKO", "HAAKO": "RUNE HAAKO",
    "DOFINE": "DOFINE", "TEY HOW": "TEY HOW", "OOM-9": "OOM-9",
    "POGGLE": "POGGLE", "WAT TAMBOR": "WAT TAMBOR", "SAN HILL": "SAN HILL",
    "SHU MAI": "SHU MAI", "PASSEL ARGENTE": "PASSEL ARGENTE", "SUN FAC": "SUN FAC",
    "LOTT DOD": "LOTT DOD", "MAS AMEDDA": "MAS AMEDDA", "MAD AMEDDA": "MAS AMEDDA",
    # --- unaligned, kept for the dataset but never labelled --------------------
    "WATTO": "WATTO", "SEBULBA": "SEBULBA", "JABBA": "JABBA", "GREEDO": "GREEDO",
    "JANGO FETT": "JANGO FETT", "BOBA FETT": "BOBA FETT",
    "ZAM WESSEL": "ZAM WESELL", "ZAM WESELL": "ZAM WESELL", "ZAM": "ZAM WESELL",
    "LAMA SU": "LAMA SU", "TAUN WE": "TAUN WE", "JIRA": "JIRA", "WALD": "WALD",
    "TC-14": "TC-14", "TC14": "TC-14",
}

#: The Jedi/Republic camp. Membership is stable across all three episodes.
LIGHT: frozenset[str] = frozenset({
    "QUI-GON", "OBI-WAN", "YODA", "MACE WINDU", "KI-ADI-MUNDI", "PLO KOON", "AAYLA",
    "JOCASTA NU", "PADME", "SHMI", "JAR JAR", "BAIL ORGANA", "MON MOTHMA", "PANAKA",
    "TYPHO", "SIO BIBBLE", "C-3PO", "VALORUM", "RIC OLIE", "BOSS NASS", "TARPALS",
    "GENERAL CEEL", "JAMILLIA", "DEXTER JETTSTER", "SOLA", "JOBAL", "RUWEE",
    "OWEN", "BERU", "CLIEGG", "DORME", "RABE", "SABE", "KITSTER", "ANTILLES",
})

#: The Sith/Separatist camp. Palpatine is here for the whole trilogy: he is a Sith
#: Lord from his first line onward, and his Chancellor register is part of what the
#: model has to learn, not a reason to relabel him.
DARK: frozenset[str] = frozenset({
    "PALPATINE", "DARTH MAUL", "DOOKU", "GRIEVOUS", "NUTE GUNRAY", "RUNE HAAKO",
    "DOFINE", "TEY HOW", "OOM-9", "POGGLE", "WAT TAMBOR", "SAN HILL", "SHU MAI",
    "PASSEL ARGENTE", "SUN FAC", "LOTT DOD", "MAS AMEDDA",
})

#: Deliberately unlabelled, with the reason. These characters oppose or help the
#: heroes without belonging to either rhetorical camp; labelling them would teach
#: the model that "criminal" and "dark side" are the same register. The original
#: project put most of them in the `mechants` list.
UNALIGNED: dict[str, str] = {
    "WATTO": "slaver and junk dealer, no faction",
    "SEBULBA": "podracing rival",
    "JABBA": "crime lord, hostile to both camps",
    "GREEDO": "child bully on Tatooine",
    "JANGO FETT": "mercenary paid by the Separatists",
    "BOBA FETT": "child, lines are grief not ideology",
    "ZAM WESELL": "contract bounty hunter",
    "LAMA SU": "Kaminoan cloner, believes he serves the Republic",
    "TAUN WE": "Kaminoan cloner, same",
    "JIRA": "Tatooine bystander",
    "WALD": "Tatooine child",
    "TC-14": "protocol droid, no allegiance",
}

#: Cues that are scene headings, titles or crowd noise rather than characters.
NOT_A_CHARACTER: frozenset[str] = frozenset({
    "INTERIOR", "EXTERIOR", "INT", "EXT", "INT.", "EXT.", "TITLE CARD", "STAR WARS",
    "CUT TO", "DISSOLVE TO", "FADE IN", "FADE OUT", "ANGLE ON", "CLOSE ON", "PAN",
    "NOTE", "SCENE", "EPISODE", "CRAWL", "VOICE", "VOICES", "ALL", "BOTH",
    "CHILDREN", "YOUNGLINGS", "CROWD", "OPPOSITION SENATORS", "ASSEMBLY", "SENATORS",
})


def canonical(cue: str) -> str | None:
    """Map a raw speaker cue to a canonical character, or None if it is not one.

    Generic role cues (``CLONE TROOPER``, ``BATTLE DROID 3B3``, ``PILOT``) collapse to
    None: they are different people sharing a job title, so grouping their lines
    under one name would break the by-character validation split.
    """
    name = " ".join(cue.replace(".", ". ").split()).strip().upper()
    name = name.replace(". ", ".").rstrip(".") if name.startswith(("CAPT.", "GEN.", "LT.")) else name
    if name in ALIASES:
        return ALIASES[name]
    if name in NOT_A_CHARACTER or len(name) < 3:
        return None
    return None  # unknown cue -> not a tracked character


def faction(character: str) -> str | None:
    """Return 'light', 'dark' or None (subject or deliberately unlabelled)."""
    if character in LIGHT:
        return "light"
    if character in DARK:
        return "dark"
    return None
