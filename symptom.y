
%{
#include <stdio.h>

int yylex(void);
void yyerror(const char *s);

int found_brown_spots = 0;
int found_yellowing = 0;
int found_drying = 0;
int found_blight = 0;
int found_rust = 0;
int found_wilting = 0;
int found_lesions = 0;
int found_mildew = 0;
%}

%define parse.error simple

%token BROWN_SPOTS BROWN SPOTS
%token YELLOWING DRYING BLIGHT RUST
%token WILTING LESIONS MILDEW OTHER

%%

input:
      /* empty */
    | words
    ;

words:
      word
    | words word
    ;

word:
      BROWN_SPOTS { found_brown_spots = 1; }
    | BROWN       { /* Standalone brown is not enough */ }
    | SPOTS       { /* Standalone spots is generic */ }
    | YELLOWING   { found_yellowing = 1; }
    | DRYING      { found_drying = 1; }
    | BLIGHT      { found_blight = 1; }
    | RUST        { found_rust = 1; }
    | WILTING     { found_wilting = 1; }
    | LESIONS     { found_lesions = 1; }
    | MILDEW      { found_mildew = 1; }
    | OTHER       { /* Ignore unsupported words */ }
    ;

%%

void yyerror(const char *s)
{
    fprintf(stderr, "Parser error: %s\n", s);
}

int main(void)
{
    if (yyparse() != 0) {
        fprintf(stderr, "Could not parse symptom description.\n");
        return 1;
    }

    printf("brown_spots=%d\n", found_brown_spots);
    printf("yellowing=%d\n", found_yellowing);
    printf("drying=%d\n", found_drying);
    printf("blight=%d\n", found_blight);
    printf("rust=%d\n", found_rust);
    printf("wilting=%d\n", found_wilting);
    printf("lesions=%d\n", found_lesions);
    printf("mildew=%d\n", found_mildew);

    return 0;
}

