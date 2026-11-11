# Desafio final OpenMP - Coeficiente de Variação
#   make           compila tudo e cria pessoas.txt (1000 pessoas)
#   make rodar     resultado do exercício (1000 pessoas) + tempo sequencial x paralelo
#   make grande    mesmo cálculo com os dados repetidos 10000x (10 milhões de pessoas)
#   make tipos     speedup de critical x atomic x reduction x schedules
#   make limpar    apaga os executáveis
CXX      = g++
CXXFLAGS = -O2 -std=c++17 -Wall -fopenmp

all: gerar_dados cv_openmp speedup_tipos pessoas.txt

gerar_dados: gerar_dados.cpp
	$(CXX) -O2 -std=c++17 -Wall gerar_dados.cpp -o gerar_dados

cv_openmp: cv_openmp.cpp
	$(CXX) $(CXXFLAGS) cv_openmp.cpp -o cv_openmp

speedup_tipos: speedup_tipos.cpp
	$(CXX) $(CXXFLAGS) speedup_tipos.cpp -o speedup_tipos

pessoas.txt: gerar_dados
	./gerar_dados 1000 pessoas.txt

rodar: all
	./cv_openmp pessoas.txt

grande: all
	./cv_openmp pessoas.txt 10000

tipos: all
	./speedup_tipos pessoas.txt

limpar:
	rm -f gerar_dados cv_openmp speedup_tipos

.PHONY: all rodar grande tipos limpar
