# Engenharia de Software com IA — Estudos, Laboratórios e Experimentos

**Learning repository · AI engineering · LLMs · RAG · MCP · Agents**

Este repositório organiza materiais e exercícios relacionados à formação em Engenharia de Software com IA. Contém **exemplos didáticos, templates, snapshots de curso e algumas adaptações/experimentos pessoais**. A presença de um módulo aqui **não significa que todo o código foi escrito por mim ou que todos os projetos foram validados em produção**.

[Meu perfil](https://github.com/Deniscs873) · [LinkedIn](https://www.linkedin.com/in/denis-correia/)

## Organização

| Módulo | Conteúdo encontrado no repositório |
| --- | --- |
| [01 — Fundamentos de IA e LLMs](modulo01-fundamentos-de-ia-e-llms-para-programadores/) | TensorFlow.js, modelos locais, parâmetros de inferência, embeddings e Neo4j/RAG |
| [02 — Integração de APIs e LLMs](modulo02-integracao-apis-llms/) | Integração de modelos, LangChain e laboratórios RAG |
| [03 — MCP](modulo03-mcp-na-pratica/) | Ferramentas MCP, clientes, servidores e exemplos de integração |
| [04 — Agentes autônomos](modulo04-criacao-de-agentes-autonomos-novo/) | Exercícios e snapshots de um copiloto de incidentes com LangGraph |
| [05–10](modulo05-ferramentas-de-IA-para-UI-UX/) | Outros materiais e experimentos de UI, AIOps, arquitetura e governança |

Os diretórios individuais têm pré-requisitos, scripts e graus diferentes de conclusão; **não existe um comando único na raiz para executar todo o repositório**.

## Exemplos para explorar

### Experimentos pessoais identificados por nome

- [Classificação introdutória — exemplo 00 Denis](modulo01-fundamentos-de-ia-e-llms-para-programadores/exemplo-00-Denis/)
- [Protótipo de e-commerce — exemplo 01 Denis](modulo01-fundamentos-de-ia-e-llms-para-programadores/exemplo-01-ecommerce-recomendations-denis/)

Os exemplos usam dados educacionais; o protótipo de e-commerce **ainda não oferece um mecanismo de recomendação de ML avaliado**.

### Exemplos de referência e material de curso

- [Exemplo de Graph RAG com Neo4j](modulo01-fundamentos-de-ia-e-llms-para-programadores/exemplo-13-embeddings-neo4j-rag/)
- [Exercícios de MCP](modulo03-mcp-na-pratica/)
- [OpsPilot — snapshot de laboratório](modulo04-criacao-de-agentes-autonomos-novo/08-projeto-pratico-opspilot-publicado/)

Estes links indicam materiais que estão presentes para estudo, **não reivindicam autoria original nem implantação própria**. Os textos de alguns exemplos fazem referência a outros autores e seus repositórios originais.

## Como executar

1. Escolha a pasta do exemplo desejado.
2. Leia o README e o manifesto de dependências daquela pasta.
3. Copie `.env.example` para `.env` quando necessário; **nunca envie tokens ou chaves reais ao Git**.
4. Instale as dependências localmente e execute os scripts indicados na pasta.
5. Avalie resultados e limitações antes de utilizar qualquer código em outros contextos.

Não utilize `pip install -r requirements.txt` na raiz: atualmente esse arquivo não existe nela.

## Autoria, atribuição e licença

O repositório contém material educacional de terceiros. O arquivo [LICENSE.md](LICENSE.md) identifica a **Unipds Educação** como detentora dos direitos sobre os materiais sob **CC BY-NC-ND 4.0**, inclusive com restrições à redistribuição de obras derivadas.

- Consulte os autores, avisos e licenças de **cada exemplo** antes de modificar, publicar novamente ou redistribuir.
- A inclusão de um arquivo aqui **não concede direitos adicionais** nem substitui a licença de origem.
- Exemplos nomeados, modificações pessoais e trabalhos próprios precisam ter autoria e proveniência identificadas individualmente.
- Quando houver dúvida sobre autorização para publicação, a abordagem adequada é **obter permissão ou retirar o material afetado**, sem assumir licenciamento permissivo.

## Padrão para novos projetos autorais

Para portfólio profissional, novos projetos independentes devem conter: escopo e problema, arquitetura, configuração reproduzível, testes, métricas, limitações, origem de dados e licença apropriada.

### Roadmap pessoal (não implementações concluídas)

- [ ] Assistente RAG autoral com citações e conjunto de avaliação
- [ ] Detecção de anomalias em dados industriais com métricas
- [ ] Pipeline de MLOps com CI/CD e monitoramento

---

## English summary

AI software engineering learning repository with course references, labs, and personal exercises. **Do not assume that every example is original work by this account's owner.** Course licensing and per-project provenance apply. The roadmap describes proposed original projects, not completed production deployments.

**Maintainer:** [Denis Correia](https://github.com/Deniscs873)
