import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl
import matplotlib.pyplot as plt

# Definição das variáveis de entrada e saída
imc = ctrl.Antecedent(np.arange(15, 41, 1), 'IMC')
atividade_fisica = ctrl.Antecedent(np.arange(0, 10, 1), 'Atividade Física')
tempo_exercicio = ctrl.Antecedent(np.arange(0, 7, 1), 'Tempo de Exercício')
tempo_atividade = ctrl.Antecedent(np.arange(0, 10, 1), 'Tempo de Atividade Física')
obesidade = ctrl.Consequent(np.arange(0, 100, 1), 'Obesidade')

# Funções de pertinência - Triangular
imc['magro'] = fuzz.trimf(imc.universe, [15, 18, 22])
imc['normal'] = fuzz.trimf(imc.universe, [18, 22, 27])
imc['sobrepeso'] = fuzz.trimf(imc.universe, [22, 27, 32])
imc['obeso'] = fuzz.trimf(imc.universe, [27, 40, 40])

atividade_fisica['baixa'] = fuzz.trimf(atividade_fisica.universe, [0, 2, 4])
atividade_fisica['moderada'] = fuzz.trimf(atividade_fisica.universe, [3, 5, 7])
atividade_fisica['alta'] = fuzz.trimf(atividade_fisica.universe, [6, 9, 9])

# Funções de pertinência - Gaussiana
tempo_exercicio['pouco'] = fuzz.gaussmf(tempo_exercicio.universe, 1, 1)
tempo_exercicio['moderado'] = fuzz.gaussmf(tempo_exercicio.universe, 3, 1)
tempo_exercicio['muito'] = fuzz.gaussmf(tempo_exercicio.universe, 5, 1)

# Funções de pertinência - Trapezoidal
tempo_atividade['baixo'] = fuzz.trapmf(tempo_atividade.universe, [0, 0, 2, 4])
tempo_atividade['médio'] = fuzz.trapmf(tempo_atividade.universe, [3, 5, 6, 8])
tempo_atividade['alto'] = fuzz.trapmf(tempo_atividade.universe, [7, 9, 10, 10])

obesidade['leve'] = fuzz.trimf(obesidade.universe, [0, 20, 40])
obesidade['moderada'] = fuzz.trimf(obesidade.universe, [30, 50, 70])
obesidade['grave'] = fuzz.trimf(obesidade.universe, [60, 80, 100])

# Definição das regras fuzzy
regra1 = ctrl.Rule(imc['magro'] & atividade_fisica['alta'] & tempo_exercicio['muito'], obesidade['leve'])
regra2 = ctrl.Rule(imc['normal'] & atividade_fisica['moderada'] & tempo_exercicio['moderado'], obesidade['moderada'])
regra3 = ctrl.Rule(imc['sobrepeso'] & atividade_fisica['baixa'] & tempo_exercicio['pouco'], obesidade['grave'])
regra4 = ctrl.Rule(imc['obeso'] & atividade_fisica['baixa'] & tempo_exercicio['pouco'], obesidade['grave'])
regra5 = ctrl.Rule(imc['obeso'] & atividade_fisica['moderada'] & tempo_exercicio['moderado'], obesidade['moderada'])
regra6 = ctrl.Rule(tempo_atividade['baixo'] & atividade_fisica['baixa'], obesidade['grave'])
regra7 = ctrl.Rule(tempo_atividade['alto'] & atividade_fisica['alta'], obesidade['leve'])

# Criando o sistema de controle fuzzy
sistema_controle = ctrl.ControlSystem([regra1, regra2, regra3, regra4, regra5, regra6, regra7])
sistema_fuzzy = ctrl.ControlSystemSimulation(sistema_controle)

# Teste com diferentes valores de entrada
valores_teste = [(18, 8, 6, 9), (25, 3, 3, 5), (30, 1, 1, 2), (35, 0, 0, 1)]
for imc_val, atividade_val, tempo_val, tempo_atv in valores_teste:
    sistema_fuzzy.input['IMC'] = imc_val
    sistema_fuzzy.input['Atividade Física'] = atividade_val
    sistema_fuzzy.input['Tempo de Exercício'] = tempo_val
    sistema_fuzzy.input['Tempo de Atividade Física'] = tempo_atv
    sistema_fuzzy.compute()
    if 'Obesidade' in sistema_fuzzy.output:
        print(f"IMC: {imc_val}, Atividade Física: {atividade_val}, Tempo de Exercício: {tempo_val}, Tempo de Atividade Física: {tempo_atv} -> Obesidade: {sistema_fuzzy.output['Obesidade']:.2f}")
    else:
        print(f"IMC: {imc_val}, Atividade Física: {atividade_val}, Tempo de Exercício: {tempo_val}, Tempo de Atividade Física: {tempo_atv} -> ERRO: Nenhuma regra cobriu esta entrada")

# Visualização das funções de pertinência
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
imc.view(ax=axs[0, 0])
atividade_fisica.view(ax=axs[0, 1])
tempo_exercicio.view(ax=axs[0, 2])
tempo_atividade.view(ax=axs[1, 0])
obesidade.view(ax=axs[1, 1])
plt.show()
