from transformers import AutoTokenizer, AutoModel
tz = AutoTokenizer.from_pretrained("codebert-base")
tz.add_tokens(['VAR'])
print(tz.convert_tokens_to_ids(["characteristically"]))

# sent = "He remains characteristically confident and optimistic."
sentenses = ['VAR1, NULL, 0, NULL, 0 )', 'char *VAR2 = FUN1( VAR1, 1 );', 'if( FUN2( VAR3, VAR4, VAR2,', 'FUN3( VAR5, FUN4( VAR2 ) );', 'FUN5( VAR2 );']
for sent in sentenses:
    print(sent)
    print(tz.tokenize(sent))
    vec = tz.convert_tokens_to_ids(tz.tokenize(sent))
    print(vec)
    print(len(vec))