# Szyfry: Cezar, Atbash, Vigenere, Podstawieniowy, Rail Fence
# Cezar
def szyfСezar(tekst, przesuniecie):
    wynik = []
    for znak in tekst:
        if znak.isalpha():
            baza = ord('A') if znak.isupper() else ord('a')
            przesuniety_kod = (ord(znak) - baza + przesuniecie) % 26 + baza
            wynik.append(chr(przesuniety_kod))
        else:
            wynik.append(znak)
    return "".join(wynik)
def deszyfСezar(szyfrogram, przesuniecie):
    return szyfСezar(szyfrogram, -przesuniecie)