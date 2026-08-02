# Phone Inventory Decision

## Choice: IPA (International Phonetic Alphabet)

## Rationale
- Our G2P tool (epitran) outputs IPA natively — using IPA directly avoids 
  an extra mapping/conversion step that could introduce errors.
- IPA is language-agnostic and widely understood by reviewers, unlike 
  IIT-M CLS which requires familiarity with that specific labeling scheme.
- Sample output confirms epitran correctly captures Tamil-specific phones 
  including retroflex consonants (ɳ, ɭ, ɻ), dental stops (n̪, t̪), and 
  long vowel length (ː) — sufficient granularity for our alignment and 
  phoneme-salience analysis needs.

## Known limitation
- If we later train the MFA acoustic model on IndicTTS, we may need to 
  verify IndicTTS's own phone labels align with IPA, or build a mapping 
  table if IndicTTS uses CLS-style labels internally.
