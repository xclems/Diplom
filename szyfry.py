# Szyfry: Cezar, Atbash, Vigenere, Podstawieniowy, Rail Fence
# Cezar
def szyfCezar(tekst, przesuniecie):
    wynik = []
    for znak in tekst:
        if znak.isalpha():
            pocz = ord('A') if znak.isupper() else ord('a')
            przesuniety_kod = (ord(znak) - pocz + przesuniecie) % 26 + pocz
            wynik.append(chr(przesuniety_kod))
        else:
            wynik.append(znak)
    return "".join(wynik)
def deszyfCezar(szyfrogram, przesuniecie):
    return szyfCezar(szyfrogram, -przesuniecie)