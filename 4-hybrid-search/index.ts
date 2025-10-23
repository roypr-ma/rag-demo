import { Database, aql } from 'arangojs';
import axios from 'axios';
import { logSection, logTime, logSeparator } from '../utils/logger.js';

// ============================================================================
// PART 4: HYBRID SEARCH WITH ARANGODB
// ============================================================================
// This demonstrates hybrid search combining:
// - BM25 keyword-based search (traditional IR)
// - Vector similarity search (semantic search)
// - Reciprocal Rank Fusion (RRF) to combine results
// ============================================================================

// ============================================================================
// CONFIGURATION
// ============================================================================

const config = {
  arango: {
    url: 'http://localhost:8529',
    password: 'testpass',
    dbName: 'hybrid_search_db',
    collectionName: 'docs',
    edgeCollectionName: 'related_to',
    graphName: 'docs_graph',
    viewName: 'docs_view',
  },
  ollama: {
    host: 'http://localhost:11434',
    model: 'nomic-embed-text',
    embeddingDimensions: 768, // nomic-embed-text produces 768-dimensional vectors
  },
};

// ============================================================================
// ARANGO & OLLAMA CLIENTS
// ============================================================================

const db = new Database({
  url: config.arango.url,
  databaseName: config.arango.dbName,
  auth: { username: 'root', password: config.arango.password },
});

const ollama = axios.create({
  baseURL: config.ollama.host,
});

// ============================================================================
// EMBEDDING FUNCTION
// ============================================================================

/**
 * Generates a vector embedding for a given text using Ollama.
 */
async function getEmbedding(text: string): Promise<number[]> {
  try {
    const response = await ollama.post('/api/embeddings', {
      model: config.ollama.model,
      prompt: text,
    });
    return response.data.embedding;
  } catch (error: any) {
    console.error(`\n❌ Error getting embedding from Ollama: ${error.message}`);
    if (error.message.includes('model not found')) {
      console.error(`\n[ACTION REQUIRED]`);
      console.error(`Please pull the model first by running:`);
      console.error(`docker exec -it ollama-server ollama pull ${config.ollama.model}\n`);
    }
    process.exit(1);
  }
}

// ============================================================================
// SAMPLE DATA - PROFESSIONAL NETWORK
// ============================================================================

// People in a professional network with their expertise and interests
const SAMPLE_DOCS = [
  {
    _key: 'alice',
    name: 'Alice Chen',
    text: 'Senior Machine Learning Engineer specializing in large language models and vector embeddings. Passionate about semantic search and RAG systems. 8 years of experience in ML and AI.',
    role: 'Senior ML Engineer',
    skills: ['Python', 'PyTorch', 'Vector Search', 'LLMs', 'Semantic Search'],
    yearsOfExperience: 8,
    expertiseLevel: 'Senior',
  },
  {
    _key: 'bob',
    name: 'Bob Martinez',
    text: 'Full-stack developer with expertise in graph databases and distributed systems. Loves building scalable backend architectures. 6 years building production systems.',
    role: 'Backend Developer',
    skills: ['Node.js', 'ArangoDB', 'Graph Databases', 'Microservices'],
    yearsOfExperience: 6,
    expertiseLevel: 'Mid-Level',
  },
  {
    _key: 'carol',
    name: 'Carol Williams',
    text: 'Data Scientist focusing on natural language processing and information retrieval. Experienced with BM25 and ranking algorithms. 5 years in NLP and search.',
    role: 'Data Scientist',
    skills: ['NLP', 'Information Retrieval', 'BM25', 'Python'],
    yearsOfExperience: 5,
    expertiseLevel: 'Mid-Level',
  },
  {
    _key: 'david',
    name: 'David Kim',
    text: 'DevOps Engineer managing cloud infrastructure and containerized applications. Expert in Docker, Kubernetes, and CI/CD pipelines. 7 years in DevOps.',
    role: 'DevOps Engineer',
    skills: ['Docker', 'Kubernetes', 'AWS', 'Terraform'],
    yearsOfExperience: 7,
    expertiseLevel: 'Senior',
  },
  {
    _key: 'emma',
    name: 'Emma Thompson',
    text: 'Principal Research Scientist working on neural search and embedding models. Published 15+ papers on hybrid search techniques combining traditional and modern approaches. 12 years in ML research, recognized expert in neural embeddings.',
    role: 'Principal Research Scientist',
    skills: ['Research', 'Neural Networks', 'Embeddings', 'Academic Writing', 'Hybrid Search'],
    yearsOfExperience: 12,
    expertiseLevel: 'Expert',
  },
  {
    _key: 'frank',
    name: 'Frank Rodriguez',
    text: 'Database Administrator with deep knowledge of multi-model databases. Specializes in query optimization and indexing strategies. 10 years managing enterprise databases.',
    role: 'Senior DBA',
    skills: ['Database Design', 'Query Optimization', 'Indexing', 'Performance Tuning'],
    yearsOfExperience: 10,
    expertiseLevel: 'Senior',
  },
  {
    _key: 'grace',
    name: 'Grace Lee',
    text: 'Product Manager for AI-powered search products. Bridging technical implementation with user needs and business requirements. 6 years in product management.',
    role: 'Product Manager',
    skills: ['Product Strategy', 'User Research', 'AI Products', 'Agile'],
    yearsOfExperience: 6,
    expertiseLevel: 'Mid-Level',
  },
  {
    _key: 'henry',
    name: 'Henry Patel',
    text: 'Principal Software Architect designing enterprise search solutions. Advocates for combining semantic understanding with traditional keyword matching. 15 years architecting large-scale systems.',
    role: 'Principal Architect',
    skills: ['System Design', 'Search Architecture', 'Scalability', 'Documentation'],
    yearsOfExperience: 15,
    expertiseLevel: 'Expert',
  },
  {
    _key: 'iris',
    name: 'Iris Wang',
    text: 'Frontend Developer creating intuitive search interfaces. Passionate about user experience and real-time search suggestions. 4 years in frontend development.',
    role: 'Frontend Developer',
    skills: ['React', 'TypeScript', 'UI/UX', 'Performance'],
    yearsOfExperience: 4,
    expertiseLevel: 'Mid-Level',
  },
  {
    _key: 'jack',
    name: 'Jack Brown',
    text: 'Technical Writer and developer advocate. Creates documentation and tutorials for complex search systems and AI tools. 5 years in technical communications.',
    role: 'Developer Advocate',
    skills: ['Technical Writing', 'Developer Relations', 'Teaching', 'Content Creation'],
    yearsOfExperience: 5,
    expertiseLevel: 'Mid-Level',
  },
];

// Professional relationships and collaborations
const SAMPLE_RELATIONSHIPS = [
  // Alice (ML Engineer) connections
  { _from: 'docs/alice', _to: 'docs/emma', type: 'collaborates_with', project: 'Neural search research' },
  { _from: 'docs/alice', _to: 'docs/carol', type: 'works_with', project: 'Hybrid ranking systems' },
  { _from: 'docs/alice', _to: 'docs/henry', type: 'consults_with', project: 'Search architecture design' },
  
  // Bob (Backend) connections
  { _from: 'docs/bob', _to: 'docs/frank', type: 'works_with', project: 'Database optimization' },
  { _from: 'docs/bob', _to: 'docs/david', type: 'collaborates_with', project: 'Infrastructure deployment' },
  { _from: 'docs/bob', _to: 'docs/henry', type: 'reports_to', project: 'Backend team' },
  
  // Carol (Data Scientist) connections
  { _from: 'docs/carol', _to: 'docs/emma', type: 'collaborates_with', project: 'Research papers' },
  { _from: 'docs/carol', _to: 'docs/alice', type: 'mentors', project: 'ML best practices' },
  
  // Emma (Research) connections
  { _from: 'docs/emma', _to: 'docs/jack', type: 'works_with', project: 'Research documentation' },
  
  // Frank (DBA) connections
  { _from: 'docs/frank', _to: 'docs/henry', type: 'advises', project: 'Database strategy' },
  
  // Grace (PM) connections
  { _from: 'docs/grace', _to: 'docs/alice', type: 'manages', project: 'ML features roadmap' },
  { _from: 'docs/grace', _to: 'docs/iris', type: 'manages', project: 'UI/UX improvements' },
  { _from: 'docs/grace', _to: 'docs/henry', type: 'works_with', project: 'Product strategy' },
  
  // Henry (Architect) connections
  { _from: 'docs/henry', _to: 'docs/jack', type: 'collaborates_with', project: 'Architecture docs' },
  
  // Iris (Frontend) connections
  { _from: 'docs/iris', _to: 'docs/bob', type: 'works_with', project: 'API integration' },
  { _from: 'docs/iris', _to: 'docs/jack', type: 'collaborates_with', project: 'UI documentation' },
  
  // Jack (DevRel) connections
  { _from: 'docs/jack', _to: 'docs/david', type: 'works_with', project: 'Deployment guides' },
];

// ============================================================================
// DATABASE SETUP
// ============================================================================

/**
 * Sets up the entire database structure:
 * 1. Creates the database
 * 2. Creates the collection
 * 3. Creates a Vector Index (for semantic search)
 * 4. Creates an ArangoSearch View (for keyword search)
 * 5. Generates embeddings for and ingests sample data
 */
async function setupDatabase() {
  logSection('📦 DATABASE SETUP');
  const startTime = Date.now();

  // 1. Create Database
  const systemDb = new Database({
    url: config.arango.url,
    databaseName: '_system',
    auth: { username: 'root', password: config.arango.password },
  });

  try {
    await systemDb.createDatabase(config.arango.dbName);
    console.log(`✓ Database '${config.arango.dbName}' created`);
  } catch (e: any) {
    if (e.isArangoError && e.errorNum === 1207) {
      console.log(`✓ Database '${config.arango.dbName}' already exists`);
    } else {
      throw e;
    }
  }

  // 2. Create Document Collection
  const col = db.collection(config.arango.collectionName);
  try {
    await db.createCollection(config.arango.collectionName);
    console.log(`✓ Document collection '${config.arango.collectionName}' created`);
  } catch (e: any) {
    if (e.isArangoError && e.errorNum === 1207) {
      console.log(`✓ Document collection '${config.arango.collectionName}' already exists`);
      await (col as any).truncate();
      console.log('  → Collection truncated');
    } else {
      throw e;
    }
  }

  // 3. Create Edge Collection
  const edgeCol = db.collection(config.arango.edgeCollectionName);
  try {
    await db.createEdgeCollection(config.arango.edgeCollectionName);
    console.log(`✓ Edge collection '${config.arango.edgeCollectionName}' created`);
  } catch (e: any) {
    if (e.isArangoError && e.errorNum === 1207) {
      console.log(`✓ Edge collection '${config.arango.edgeCollectionName}' already exists`);
      await (edgeCol as any).truncate();
      console.log('  → Edge collection truncated');
    } else {
      throw e;
    }
  }

  // 4. Create Named Graph
  try {
    await db.createGraph(config.arango.graphName, [
      {
        collection: config.arango.edgeCollectionName,
        from: [config.arango.collectionName],
        to: [config.arango.collectionName],
      },
    ]);
    console.log(`✓ Graph '${config.arango.graphName}' created`);
  } catch (e: any) {
    if (e.isArangoError && e.errorNum === 1207) {
      console.log(`✓ Graph '${config.arango.graphName}' already exists`);
    } else {
      throw e;
    }
  }

  // 5. Create ArangoSearch View (for BM25)
  try {
    await db.createView(config.arango.viewName, {
      type: 'arangosearch',
      links: {
        [config.arango.collectionName]: {
          fields: {
            text: { analyzers: ['text_en'] },
          },
        },
      },
    });
    console.log(`✓ ArangoSearch view '${config.arango.viewName}' created`);
  } catch (e: any) {
    if (e.isArangoError && e.errorNum === 1207) {
      console.log(`✓ ArangoSearch view '${config.arango.viewName}' already exists`);
    } else {
      throw e;
    }
  }

  // 6. Ingest sample documents
  console.log(`\n📝 Ingesting ${SAMPLE_DOCS.length} sample documents...`);
  for (const doc of SAMPLE_DOCS) {
    console.log(`   → Embedding "${doc._key}"...`);
    const embedding = await getEmbedding(doc.text);
    await (col as any).save({
      ...doc,
      embedding: embedding,
    });
  }

  // 7. Create relationships (edges)
  console.log(`\n🔗 Creating ${SAMPLE_RELATIONSHIPS.length} relationships...`);
  for (const edge of SAMPLE_RELATIONSHIPS) {
    await (edgeCol as any).save(edge);
  }
  console.log(`   ✓ Graph relationships created`);

  // 8. Create Vector Index (after documents are inserted)
  // Vector indexes enable fast similarity search using Approximate Nearest Neighbor (ANN)
  try {
    await (col as any).ensureIndex({
      // type: 'vector' - Specifies this is a vector index for similarity search
      // Uses IVF (Inverted File) algorithm for fast approximate nearest neighbor search
      type: 'vector',
      
      // name: 'idx_vector' - Unique identifier for this index
      // Used to reference the index in queries and management operations
      name: 'idx_vector',
      
      // fields: ['embedding'] - The document field containing the vector
      // Must be an array of numbers (e.g., [0.23, -0.15, 0.42, ...])
      fields: ['embedding'],
      
      params: {
        // metric: 'cosine' - Distance metric for similarity calculation
        // Options: 'cosine', 'euclidean' (L2), 'manhattan' (L1)
        // Cosine is best for normalized vectors (direction matters, magnitude doesn't)
        metric: 'cosine',
        
        // dimension: 768 - Number of dimensions in each vector
        // MUST match the embedding model output (nomic-embed-text = 768)
        // Mismatch will cause errors during search
        dimension: config.ollama.embeddingDimensions,
        
        // nLists: 1 - Number of inverted lists (partitions) for IVF algorithm
        // Higher = faster search but less accurate, lower = slower but more accurate
        // Rule of thumb: sqrt(num_documents) or 1 for small datasets (<10k docs)
        nLists: 1,
      },
      
      // inBackground: false - Index creation blocks until complete
      // true = non-blocking (faster) but index not immediately available
      // false = blocking (safer) ensures index ready before proceeding
      inBackground: false,
    });
    console.log(`✓ Vector index 'idx_vector' created`);
  } catch (e: any) {
    if (e.isArangoError && e.errorNum === 1210) {
      console.log(`✓ Vector index 'idx_vector' already exists`);
    } else {
      console.error('Failed to create vector index:', e.message);
      throw e;
    }
  }

  const duration = (Date.now() - startTime) / 1000;
  console.log(`\n✓ Setup complete`);
  logTime(duration);
  logSeparator();
}

// ============================================================================
// DATABASE CHECK FUNCTION
// ============================================================================

/**
 * Checks if the database is set up by trying to access the collection
 */
async function isDatabaseSetup(): Promise<boolean> {
  try {
    const col = db.collection(config.arango.collectionName);
    const count = await (col as any).count();
    return count.count > 0;
  } catch (e: any) {
    return false;
  }
}

// ============================================================================
// MULTI-MODEL HYBRID SEARCH
// ============================================================================

/**
 * Performs multi-model search combining:
 * 1. BM25 keyword search (via ArangoSearch view)
 * 2. Vector similarity search (via vector index)
 * 3. Reciprocal Rank Fusion (RRF) to merge results
 * 4. Graph traversal to include connected people
 */
async function search(query: string) {

  logSection('🔍 MULTI-MODEL SEARCH (Hybrid + Graph)');
  console.log(`Query: "${query}"\n`);

  const startTime = Date.now();

  // 1. Generate query embedding
  console.log('📊 Generating query embedding...');
  const queryVector = await getEmbedding(query);
  console.log(`   ✓ Generated ${queryVector.length}-dimensional vector\n`);

  // 2. Execute Multi-Model Search
  console.log('🔄 Executing multi-model search...');
  console.log('   → BM25 keyword search');
  console.log('   → Vector semantic search'); 
  console.log('   → RRF fusion');
  console.log('   → Graph expansion\n');
  
  const k = 60; // RRF constant
  const limit = 3; // Initial results, then expand via graph

  const aqlQuery = aql`
    LET k = ${k}
    LET query_text = ${query}
    LET query_vector = ${queryVector}

    // 1. BM25 Keyword Search
    LET bm25_docs = (
        FOR doc IN ${db.view(config.arango.viewName)}
        SEARCH ANALYZER(doc.text IN TOKENS(query_text, 'text_en'), 'text_en')
        LET bm25_score = BM25(doc)
        SORT bm25_score DESC
        LIMIT ${limit}
        RETURN doc
    )

    // 2. Vector Similarity Search
    LET vector_docs = (
        FOR doc IN ${db.collection(config.arango.collectionName)}
        LET similarity = COSINE_SIMILARITY(doc.embedding, query_vector)
        SORT similarity DESC
        LIMIT ${limit}
        RETURN doc
    )

    // 3. Apply RRF scoring
    LET bm25_with_rrf = (
        FOR i IN 0..LENGTH(bm25_docs)-1
        LET doc = bm25_docs[i]
        LET rank = i + 1
        RETURN {
            _id: doc._id,
            _key: doc._key,
            name: doc.name,
            text: doc.text,
            role: doc.role,
            yearsOfExperience: doc.yearsOfExperience,
            expertiseLevel: doc.expertiseLevel,
            rrf_score: 1.0 / (k + rank),
            source: "hybrid_search"
        }
    )

    LET vector_with_rrf = (
        FOR i IN 0..LENGTH(vector_docs)-1
        LET doc = vector_docs[i]
        LET rank = i + 1
        RETURN {
            _id: doc._id,
            _key: doc._key,
            name: doc.name,
            text: doc.text,
            role: doc.role,
            yearsOfExperience: doc.yearsOfExperience,
            expertiseLevel: doc.expertiseLevel,
            rrf_score: 1.0 / (k + rank),
            source: "hybrid_search"
        }
    )

    // 4. Get initial hybrid results
    LET initial_results = (
        FOR doc IN UNION(bm25_with_rrf, vector_with_rrf)
        COLLECT _id = doc._id, _key = doc._key, name = doc.name, text = doc.text, role = doc.role,
                yearsOfExperience = doc.yearsOfExperience, expertiseLevel = doc.expertiseLevel
        AGGREGATE final_score = SUM(doc.rrf_score)
        SORT final_score DESC
        LIMIT ${limit}
        RETURN {
            _id: _id,
            _key: _key,
            name: name,
            text: text,
            role: role,
            yearsOfExperience: yearsOfExperience,
            expertiseLevel: expertiseLevel,
            score: final_score,
            source: "hybrid_search"
        }
    )

    // 5. GRAPH EXPANSION: Get connected people
    LET related_people = (
        FOR result IN initial_results
        FOR v, e, p IN 1..1 ANY result._id GRAPH ${config.arango.graphName}
            RETURN DISTINCT {
                _id: v._id,
                _key: v._key,
                name: v.name,
                text: v.text,
                role: v.role,
                yearsOfExperience: v.yearsOfExperience,
                expertiseLevel: v.expertiseLevel,
                score: 0.005,
                source: "graph_expansion",
                relationship: e.type,
                connected_to: result._key,
                project: e.project
            }
    )

    // 6. Combine direct matches with graph-expanded results
    FOR person IN UNION_DISTINCT(initial_results, related_people)
    SORT person.score DESC, person.source ASC
    RETURN person
  `;

  const cursor = await db.query(aqlQuery);
  const results = await cursor.all();
  const duration = (Date.now() - startTime) / 1000;

  // 3. Display results with graph relationships
  console.log(`   ✓ Search complete (${results.length} people found)\n`);
  
  logSection('📋 RESULTS');
  
  const directMatches = results.filter((r: any) => r.source === 'hybrid_search');
  const graphMatches = results.filter((r: any) => r.source === 'graph_expansion');
  
  console.log(`\n🎯 Direct Matches (${directMatches.length}):`);
  directMatches.forEach((person: any, i: number) => {
    const expertBadge = person.expertiseLevel === 'Expert' ? ' ⭐ EXPERT' : person.expertiseLevel === 'Senior' ? ' 🔹 Senior' : '';
    console.log(`\n${i + 1}. ${person.name} - ${person.role}${expertBadge}`);
    console.log(`   Experience: ${person.yearsOfExperience} years | Match Score: ${person.score.toFixed(4)}`);
    console.log(`   ${person.text}`);
  });

  if (graphMatches.length > 0) {
    console.log(`\n\n🔗 Connected People (${graphMatches.length}):`);
    graphMatches.forEach((person: any, i: number) => {
      const expertBadge = person.expertiseLevel === 'Expert' ? ' ⭐ EXPERT' : person.expertiseLevel === 'Senior' ? ' 🔹 Senior' : '';
      console.log(`\n${i + 1}. ${person.name} - ${person.role}${expertBadge}`);
      console.log(`   Experience: ${person.yearsOfExperience} years | Via: ${person.relationship} ${person.connected_to}`);
      console.log(`   Project: "${person.project}"`);
      console.log(`   ${person.text}`);
    });
  }

  console.log();
  logTime(duration);
  logSeparator();
}

// ============================================================================
// RESET FUNCTION
// ============================================================================

/**
 * Drops the database to start fresh
 */
async function resetDatabase() {
  logSection('🗑️  DATABASE RESET');
  
  try {
    const systemDb = new Database({
      url: config.arango.url,
      databaseName: '_system',
      auth: { username: 'root', password: config.arango.password },
    });

    await systemDb.dropDatabase(config.arango.dbName);
    console.log(`✓ Database '${config.arango.dbName}' dropped successfully\n`);
    console.log('Run search again to recreate the database automatically.\n');
  } catch (e: any) {
    if (e.isArangoError && e.errorNum === 1228) {
      console.log(`⚠️  Database '${config.arango.dbName}' does not exist (already clean)\n`);
    } else {
      console.error('❌ Error dropping database:', e.message);
      throw e;
    }
  }
  
  logSeparator();
}

// ============================================================================
// MAIN FUNCTION
// ============================================================================

async function main() {
  const command = process.argv[2];

  // Handle reset command
  if (command === 'reset' || command === '--reset') {
    await resetDatabase();
    return;
  }

  const query = command;

  if (!query) {
    console.log('\n❌ No search query provided.\n');
    console.log('Usage:');
    console.log('  yarn start:hybrid "your query"        - Search the knowledge base');
    console.log('  yarn start:hybrid reset               - Drop database and start fresh\n');
    console.log('Search Examples:');
    console.log('  yarn start:hybrid "help building search with neural embeddings"\n');
    console.log('  This query demonstrates all 3 search types:');
    console.log('    • BM25 finds exact keywords ("neural", "embeddings")');
    console.log('    • Vector understands semantic meaning (search systems, ML expertise)');
    console.log('    • Graph discovers collaborators (people who work together)\n');
    console.log('This searches a social network knowledge base using:');
    console.log('  • BM25 keyword matching (exact terms)');
    console.log('  • Vector semantic similarity (meaning & context)');
    console.log('  • RRF fusion (combines both)');
    console.log('  • Graph expansion (finds collaborators)\n');
    return;
  }

  // Check if database is set up
  const isSetup = await isDatabaseSetup();
  
  if (!isSetup) {
    console.log('🔧 Knowledge base not found. Setting up first...\n');
    await setupDatabase();
    console.log();
  }

  // Run multi-model search
  await search(query);
}

main().catch((e: any) => {
  console.error('\n❌ An unhandled error occurred:');
  console.error(e);
  process.exit(1);
});

