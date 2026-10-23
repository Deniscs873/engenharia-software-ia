# Engenharia de Software com Inteligência Artificial

Repositório técnico voltado ao desenvolvimento, arquitetura e implementação de pipelines práticos de IA. Reúne projetos autorais, estudos de caso e desafios de engenharia construídos ao longo da minha especialização em IA Aplicada, demonstrando evolução contínua em arquitetura de software, orquestração de LLMs e automação inteligente.

---

## 🎯 Proposta & Foco Técnico

Diferente de um repositório convencional de exercícios acadêmicos, este espaço documenta a **minha curva de evolução técnica** e a transição entre consumir APIs simples de linguagem e desenhar sistemas autônomos, resilientes e orientados a grafos.

* **Foco em Boas Práticas:** Separação estrita de responsabilidades, tipagem estática com Pydantic, controle determinístico de respostas e tratamento de falhas em chamadas a modelos probabilísticos.
* **Execução Híbrida:** Implementações divididas entre modelos de fronteira em nuvem e inferência local (*on-premise / edge*) para privacidade e contenção de custos.
* **Engenharia de Recuperação (RAG):** Evolução de RAG puramente vetorial para Graph RAG via Neo4j, reduzindo alucinações em dados estruturados complexos.

---

## 📈 Trilha de Evolução & Roadmap Prático

| Fase / Módulo | Domínio Principal | Tecnologias Chave | Status |
| :--- | :--- | :--- | :--- |
| **01. Fundamentos & Core LLMs** | Parâmetros de inferência, Multimodalidade e Graph RAG | Python, Ollama, Neo4j, Playwright | Concluído |
| **02. Integração de APIs** | Padrões de consumo resiliente, Schema Enforcement | OpenRouter, Pydantic, REST | Concluído |
| **03. MCP na Prática** | Model Context Protocol, integração de ferramentas e dados | MCP SDK, Grafana, Stdio/SSE | Em andamento |
| **04. Agentes Autônomos** | Loops ReAct, planejamento de tarefas e memória de contexto | LangGraph / CrewAI, Memória Persistente | Em andamento |
| **05 a 07. Interface & Gestão** | Design de interfaces orientadas a IA e fluxos AIOps | Streamlit, UI/UX com IA, Métricas | Planejado |
| **08 a 10. Arquitetura & Governança** | Fine-tuning, Observabilidade, Segurança e Conformidade | LoRA/QLoRA, Grafana, Guardrails | Planejado |

---

## 💡 Destaques de Engenharia (Case Studies)

Abaixo estão alguns dos desafios reais implementados no repositório:

### 1. Graph RAG com Neo4j & Embeddings (`modulo01`)
* **Problema:** Sistemas tradicionais de RAG vetorial perdem conexões semânticas relacionais em conjuntos de dados densos.
* **Solução:** Implementação de pipeline combinando busca por similaridade vetorial com navegação em grafos relacionais no Neo4j, garantindo recuperação de contexto hierárquico e redução de alucinações.
* **Stack:** Neo4j, Cypher, Embeddings semânticos, Python.

### 2. Padronização de Ferramentas com MCP (`modulo03`)
* **Problema:** Conexão direta e desordenada entre agentes e fontes de dados legadas.
* **Solução:** Adoção do *Model Context Protocol* (MCP) para padronizar a exposição de métricas do Grafana e contexto local de arquivos para os modelos de linguagem, criando uma camada de desacoplamento.
* **Stack:** Protocolo MCP, Grafana, JSON-RPC.

---

## 🛠️ Stack Tecnológica Consolidada

* **Linguagens & Ambientes:** Python 3.11+, Node.js, Ambientes Virtuais (.venv isolados).
* **LLMs & Inferência:** Ollama (execução local / zero-cost inference), OpenRouter, Claude, GPT.
* **Bancos de Dados & Grafos:** Neo4j (Cypher), Armazenamento de Vetores.
* **Observabilidade & Validação:** Grafana, Pydantic, Playwright (testes e raspagem orientada a agentes).

---

## 📓 Diário de Bordo (DevLog & Decisões de Arquitetura)

Registro conciso das decisões e aprendizados práticos adquiridos durante as implementações:

* **Controle de Variabilidade em Inferência:** Nos primeiros testes, o uso de `temperature` padrão gerava variabilidade excessiva em extrações de dados. A padronização com amostragem baixa ($temperature \le 0.2$) combinada com schemas estritos garantiu deterministicidade necessária para integração com APIs.
* **Eficiência com Modelos Locais:** A integração do Ollama permitiu construir baterias de testes automatizados sem custo de tokens de APIs comerciais, reservando modelos de fronteira apenas para tarefas que exigem alto poder de raciocínio.
* **Protocolo MCP vs. Chamadas de Função Tradicionais:** A transição para MCP permitiu que ferramentas de contexto (como métricas e logs) pudessem ser reaproveitadas por múltiplos agentes sem a necessidade de reescrever wrappers de chamadas proprietárias.

---

## 🚀 Como Explorar e Executar

### 1. Clonar o repositório
```bash
git clone https://github.com/Deniscs873/engenharia-software-ia.git
cd engenharia-software-ia
```

### 2. Ativar o ambiente virtual
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instalar dependências
```bash
pip install -r requirements.txt
```

---

## 👨‍💻 Autor

**Denis Correia**  
Engenheiro & Desenvolvedor de Software  
* Perfil GitHub: [@Deniscs873](https://github.com/Deniscs873)  
* Foco: Arquitetura de Sistemas com IA, Sistemas Embarcados/IoT e Engenharia de Software.