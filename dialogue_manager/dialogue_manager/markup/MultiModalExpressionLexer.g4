// Grammar for multi-modal expressions -- REFERENCE ONLY; the parser in parser.py is fully stand-alone, and does not rely on this grammar
lexer grammar MultiModalExpressionLexer;

// Default mode
A_OPEN  : '<' -> pushMode(ACTION) ;
VT_OPEN : '@' -> type(AT), pushMode(VARIABLE_TEXT) ;
TEXT    : ~[<@]+ ;

// Action mode: everything inside of angle brackets
mode ACTION ;

A_CLOSE : '>' -> popMode ;

DO      : 'do' ;
FALSE   : 'false' | 'False' ;
NULL    : 'null' ;
PAUSE   : 'pause' ;
SET     : 'set' ;
START   : 'start' ;
STOP    : 'stop' ;
TIMEOUT : 'timeout' ;
TRUE    : 'true' | 'True' ;
TTS     : 'tts' ;
WAIT    : 'wait' ;

LPAREN  : '(' ;
RPAREN  : ')' ;
LBRACE  : '[' ;
RBRACE  : ']' ;  
DOT     : '.' ;
COMMA   : ',' ;
ASSIGN  : '=' ;
AT      : '@' ;
COLUMN  : '|' ;

STRING  : '"' DOUBLE_QUOTE_CHAR* '"'
        | '\'' SINGLE_QUOTE_CHAR* '\'' ;
fragment DOUBLE_QUOTE_CHAR  : ~["\\] | '\\"' ;   // not " or \ unless \"
fragment SINGLE_QUOTE_CHAR  : ~['\\] | '\\\'' ;  // not ' or \ unless \'

NUMBER  : SIGN? INT ('.' [0-9]*)? EXP?  // +1.e2, 1234, 1234.5
        | SIGN? '.' [0-9]+ EXP?         // -.2e3
        | SIGN? '0' [xX] HEX+           // 0x12345678
        ;
fragment SIGN   : [+-] ;
fragment HEX    : [0-9a-fA-F] ;
fragment INT    : '0' | [1-9] [0-9]*  ;
fragment EXP    : [Ee] SIGN? [0-9]* ;

ID                  : (ID_HEAD ID_BODY*) ;
fragment ID_HEAD    : [a-zA-Z] ;
fragment ID_BODY    : ID_HEAD | [0-9_] ;

WS      : [ \t\r\n] -> skip ;

// Variable text mode: everything inside {@ ... }
mode VARIABLE_TEXT ;

DT_COLUMN   : '|' -> type(COLUMN) ;
DT_DOT      : '.' -> type(DOT) ;
DT_STRING   : STRING -> type(STRING), popMode ;
DT_ID       : ID -> type(ID) ;
DT_WS       : WS -> skip ;
