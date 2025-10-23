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
    projectCollectionName: 'projects',
    edgeCollections: {
      reports_to: 'reports_to',
      collaborates_with: 'collaborates_with',
      works_with: 'works_with',
      advises: 'advises',
      works_on: 'works_on',
    },
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

// People in a professional network - Search Team Hierarchy
const SAMPLE_DOCS = [
  {
    _key: 'henry',
    name: 'Henry Patel',
    text: 'VP of Engineering and Principal Architect designing enterprise search solutions. Leads the search infrastructure team. 15 years architecting large-scale systems.',
    role: 'VP Engineering',
    skills: ['System Design', 'Search Architecture', 'Leadership', 'Scalability'],
    yearsOfExperience: 15,
    expertiseLevel: 'Expert',
  },
  {
    _key: 'emma',
    name: 'Emma Thompson',
    text: 'Principal Research Scientist working on neural search and embedding models. Published 15+ papers on hybrid search techniques. Reports to Henry. 12 years in ML research, recognized expert in neural embeddings.',
    role: 'Principal Research Scientist',
    skills: ['Research', 'Neural Networks', 'Embeddings', 'Hybrid Search'],
    yearsOfExperience: 12,
    expertiseLevel: 'Expert',
  },
  {
    _key: 'alice',
    name: 'Alice Chen',
    text: 'Senior Machine Learning Engineer specializing in semantic search and vector embeddings. Works under Emma on implementing search features. 8 years of experience in ML and AI.',
    role: 'Senior ML Engineer',
    skills: ['Python', 'PyTorch', 'Vector Search', 'LLMs', 'Semantic Search'],
    yearsOfExperience: 8,
    expertiseLevel: 'Senior',
  },
  {
    _key: 'bob',
    name: 'Bob Martinez',
    text: 'Backend Engineer building search APIs and integrations. Reports to Henry. Implements graph databases and distributed search systems. 6 years building production systems.',
    role: 'Backend Engineer',
    skills: ['Node.js', 'ArangoDB', 'Graph Databases', 'APIs'],
    yearsOfExperience: 6,
    expertiseLevel: 'Mid-Level',
  },
  {
    _key: 'carol',
    name: 'Carol Williams',
    text: 'Data Scientist working on search relevance and ranking. Collaborates with Emma on BM25 and hybrid search algorithms. 5 years in NLP and information retrieval.',
    role: 'Data Scientist',
    skills: ['NLP', 'Information Retrieval', 'BM25', 'Ranking Algorithms'],
    yearsOfExperience: 5,
    expertiseLevel: 'Mid-Level',
  },
];

// Projects - These become nodes in the graph
const SAMPLE_PROJECTS = [
  {
    _key: 'hybrid_search',
    name: 'Hybrid Search Implementation',
    description: 'Building a hybrid search system combining BM25, vector similarity, and graph traversal',
    status: 'active',
  },
  {
    _key: 'search_api',
    name: 'Search API Integration',
    description: 'REST API for search functionality with ArangoDB backend',
    status: 'active',
  },
  {
    _key: 'neural_embeddings',
    name: 'Neural Embeddings Research',
    description: 'Research on semantic search using neural network embeddings',
    status: 'active',
  },
];

// Relationships - Mix of person-to-person and person-to-project
const SAMPLE_RELATIONSHIPS = [
  // Reporting structure (person → person)
  { _from: 'docs/emma', _to: 'docs/henry', type: 'reports_to' },
  { _from: 'docs/bob', _to: 'docs/henry', type: 'reports_to' },
  { _from: 'docs/alice', _to: 'docs/emma', type: 'reports_to' },
  { _from: 'docs/carol', _to: 'docs/emma', type: 'reports_to' },
  
  // Cross-functional collaboration (person ↔ person)
  { _from: 'docs/alice', _to: 'docs/carol', type: 'collaborates_with' },
  { _from: 'docs/carol', _to: 'docs/alice', type: 'collaborates_with' },
  { _from: 'docs/bob', _to: 'docs/alice', type: 'works_with' },
  { _from: 'docs/emma', _to: 'docs/henry', type: 'advises' },
  
  // Project assignments (person → project)
  { _from: 'docs/alice', _to: 'projects/hybrid_search', type: 'works_on', role: 'Lead ML Engineer' },
  { _from: 'docs/carol', _to: 'projects/hybrid_search', type: 'works_on', role: 'Data Scientist' },
  { _from: 'docs/bob', _to: 'projects/search_api', type: 'works_on', role: 'Lead Backend Engineer' },
  { _from: 'docs/alice', _to: 'projects/search_api', type: 'works_on', role: 'ML Integration' },
  { _from: 'docs/emma', _to: 'projects/neural_embeddings', type: 'works_on', role: 'Principal Investigator' },
  { _from: 'docs/alice', _to: 'projects/neural_embeddings', type: 'works_on', role: 'Research Engineer' },
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

  // 2. Create Document Collections
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

  const projectCol = db.collection(config.arango.projectCollectionName);
  try {
    await db.createCollection(config.arango.projectCollectionName);
    console.log(`✓ Project collection '${config.arango.projectCollectionName}' created`);
  } catch (e: any) {
    if (e.isArangoError && e.errorNum === 1207) {
      console.log(`✓ Project collection '${config.arango.projectCollectionName}' already exists`);
      await (projectCol as any).truncate();
      console.log('  → Collection truncated');
    } else {
      throw e;
    }
  }

  // 3. Create Edge Collections (one for each relationship type)
  console.log(`\n🔗 Creating edge collections...`);
  const edgeCollections: { [key: string]: any } = {};
  for (const [type, collectionName] of Object.entries(config.arango.edgeCollections)) {
    const edgeCol = db.collection(collectionName);
    try {
      await db.createEdgeCollection(collectionName);
      console.log(`   ✓ Edge collection '${collectionName}' created`);
    } catch (e: any) {
      if (e.isArangoError && e.errorNum === 1207) {
        console.log(`   ✓ Edge collection '${collectionName}' already exists`);
        await (edgeCol as any).truncate();
        console.log(`     → Truncated`);
      } else {
        throw e;
      }
    }
    edgeCollections[type] = edgeCol;
  }

  // 4. Create Named Graph with multiple edge definitions
  try {
    const edgeDefinitions = [
      // Person-to-person relationships
      {
        collection: config.arango.edgeCollections.reports_to,
        from: [config.arango.collectionName],
        to: [config.arango.collectionName],
      },
      {
        collection: config.arango.edgeCollections.collaborates_with,
        from: [config.arango.collectionName],
        to: [config.arango.collectionName],
      },
      {
        collection: config.arango.edgeCollections.works_with,
        from: [config.arango.collectionName],
        to: [config.arango.collectionName],
      },
      {
        collection: config.arango.edgeCollections.advises,
        from: [config.arango.collectionName],
        to: [config.arango.collectionName],
      },
      // Person-to-project relationships
      {
        collection: config.arango.edgeCollections.works_on,
        from: [config.arango.collectionName],
        to: [config.arango.projectCollectionName],
      },
    ];
    
    await db.createGraph(config.arango.graphName, edgeDefinitions);
    console.log(`✓ Graph '${config.arango.graphName}' created with ${edgeDefinitions.length} edge types`);
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

  // 6. Ingest sample documents (people)
  console.log(`\n📝 Ingesting ${SAMPLE_DOCS.length} team members...`);
  for (const doc of SAMPLE_DOCS) {
    console.log(`   → Embedding "${doc._key}"...`);
    const embedding = await getEmbedding(doc.text);
    await (col as any).save({
      ...doc,
      embedding: embedding,
    });
  }

  // 6b. Ingest projects (no embeddings needed for now)
  console.log(`\n📁 Creating ${SAMPLE_PROJECTS.length} projects...`);
  for (const project of SAMPLE_PROJECTS) {
    await (projectCol as any).save(project);
    console.log(`   ✓ ${project.name}`);
  }

  // 7. Create organizational relationships (edges) in their respective collections
  console.log(`\n🔗 Creating ${SAMPLE_RELATIONSHIPS.length} organizational relationships...`);
  for (const edge of SAMPLE_RELATIONSHIPS) {
    const edgeCol = edgeCollections[edge.type];
    if (!edgeCol) {
      throw new Error(`Unknown relationship type: ${edge.type}`);
    }
    // Save edge without the type field since the collection name already indicates the type
    const { type, ...edgeData } = edge;
    await (edgeCol as any).save(edgeData);
    
    // Pretty print based on edge type
    const fromType = edge._from.startsWith('docs/') ? '👤' : '📁';
    const toType = edge._to.startsWith('docs/') ? '👤' : '📁';
    const roleSuffix = edge.role ? ` (${edge.role})` : '';
    console.log(`   ✓ ${edge.type}: ${fromType}${edge._from} → ${toType}${edge._to}${roleSuffix}`);
  }
  console.log(`   ✓ All relationships created`);

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

// Removed isDatabaseSetup() - no longer needed since we reset on every run

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

    // 4. Get initial hybrid results with their relationships
    LET initial_results = (
        FOR doc IN UNION(bm25_with_rrf, vector_with_rrf)
        COLLECT 
            _id = doc._id,
            _key = doc._key,
            name = doc.name,
            text = doc.text,
            role = doc.role,
            yearsOfExperience = doc.yearsOfExperience,
            expertiseLevel = doc.expertiseLevel
        AGGREGATE final_score = SUM(doc.rrf_score)
        FILTER _key != null
        SORT final_score DESC
        LIMIT ${limit}
        // Get relationships for this person (both to other people and to projects)
        LET person_rels = (
            FOR v, e IN 1..1 ANY _id GRAPH ${config.arango.graphName}
                // Extract relationship type from edge collection name (e.g., "reports_to/123" -> "reports_to")
                LET edge_type = SPLIT(e._id, "/")[0]
                LET node_type = SPLIT(v._id, "/")[0]
                RETURN {
                    type: edge_type,
                    connected_to: v.name,
                    connected_type: node_type,
                    role: e.role,
                    direction: e._from == _id ? "from" : "to"
                }
        )
        RETURN {
            _id: _id,
            _key: _key,
            name: name,
            text: text,
            role: role,
            yearsOfExperience: yearsOfExperience,
            expertiseLevel: expertiseLevel,
            score: final_score,
            source: "hybrid_search",
            relationships: person_rels
        }
    )

    // 5. GRAPH EXPANSION: Get connected people (excluding those already in direct matches)
    LET direct_ids = initial_results[*]._id
    LET related_people = (
        FOR result IN initial_results
        FOR v, e, p IN 1..1 ANY result._id GRAPH ${config.arango.graphName}
            FILTER v._id NOT IN direct_ids
            COLLECT person_id = v._id INTO relationships
            LET person = FIRST(relationships[*].v)
            LET rel_list = (
                FOR r IN relationships
                    // Extract relationship type from edge collection name
                    LET edge_type = SPLIT(r.e._id, "/")[0]
                    LET node_type = SPLIT(r.result._id, "/")[0]
                    RETURN {
                        type: edge_type,
                        connected_to: r.result.name,
                        connected_type: node_type,
                        role: r.e.role,
                        // If edge goes FROM this person TO the result, person is the source
                        direction: r.e._from == person_id ? "from" : "to"
                    }
            )
            RETURN {
                _id: person_id,
                _key: person._key,
                name: person.name,
                text: person.text,
                role: person.role,
                yearsOfExperience: person.yearsOfExperience,
                expertiseLevel: person.expertiseLevel,
                score: 0.005,
                source: "graph_expansion",
                relationships: rel_list
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

  // Show projects that direct matches work on
  if (directMatches.length > 0) {
    console.log(`\n\n📁 Projects:`);
    const projects = new Map<string, string[]>();
    
    for (const person of directMatches) {
      if (person.relationships) {
        for (const rel of person.relationships) {
          if (rel.connected_type === 'projects') {
            if (!projects.has(rel.connected_to)) {
              projects.set(rel.connected_to, []);
            }
            const roleInfo = rel.role ? ` (${rel.role})` : '';
            projects.get(rel.connected_to)!.push(`${person.name}${roleInfo}`);
          }
        }
      }
    }
    
    if (projects.size > 0) {
      projects.forEach((members, projectName) => {
        console.log(`   • ${projectName}`);
        members.forEach(member => console.log(`     - ${member}`));
      });
    } else {
      console.log(`   (No project connections found)`);
    }
  }

  if (graphMatches.length > 0) {
    console.log(`\n\n🔗 Connected People (${graphMatches.length}):`);
    graphMatches.forEach((person: any, i: number) => {
      const expertBadge = person.expertiseLevel === 'Expert' ? ' ⭐ EXPERT' : person.expertiseLevel === 'Senior' ? ' 🔹 Senior' : '';
      console.log(`\n${i + 1}. ${person.name} - ${person.role}${expertBadge}`);
      console.log(`   Experience: ${person.yearsOfExperience} years`);
      
      // Display all relationships
      if (person.relationships && person.relationships.length > 0) {
        console.log(`   Connected via:`);
        person.relationships.forEach((rel: any) => {
          let relationLabel = '';
          const connectedIcon = rel.connected_type === 'projects' ? '📁' : '👤';
          
          if (rel.connected_type === 'projects') {
            // Connection to a project
            const roleInfo = rel.role ? ` as ${rel.role}` : '';
            relationLabel = `Works on ${connectedIcon} ${rel.connected_to}${roleInfo}`;
          } else if (rel.direction === 'from') {
            // This person is the source of the relationship (to another person)
            relationLabel = rel.type === 'reports_to' ? `Reports to ${connectedIcon} ${rel.connected_to}` :
                           rel.type === 'collaborates_with' ? `Collaborates with ${connectedIcon} ${rel.connected_to}` :
                           rel.type === 'works_with' ? `Works with ${connectedIcon} ${rel.connected_to}` :
                           rel.type === 'advises' ? `Advises ${connectedIcon} ${rel.connected_to}` :
                           `${rel.type} → ${connectedIcon} ${rel.connected_to}`;
          } else {
            // This person is the target of the relationship (from another person)
            relationLabel = rel.type === 'reports_to' ? `${connectedIcon} ${rel.connected_to} reports to them` :
                           rel.type === 'collaborates_with' ? `Collaborates with ${connectedIcon} ${rel.connected_to}` :
                           rel.type === 'works_with' ? `${connectedIcon} ${rel.connected_to} works with them` :
                           rel.type === 'advises' ? `Receives advice from ${connectedIcon} ${rel.connected_to}` :
                           `${connectedIcon} ${rel.connected_to} → ${rel.type}`;
          }
          console.log(`     • ${relationLabel}`);
        });
      }
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
 * Drops the database to start fresh (silent operation)
 */
async function resetDatabase() {
  try {
    const systemDb = new Database({
      url: config.arango.url,
      databaseName: '_system',
      auth: { username: 'root', password: config.arango.password },
    });

    await systemDb.dropDatabase(config.arango.dbName);
  } catch (e: any) {
    // Silently ignore if database doesn't exist (error 1228)
    if (e.isArangoError && e.errorNum !== 1228) {
      console.error('❌ Error dropping database:', e.message);
      throw e;
    }
  }
}

// ============================================================================
// MAIN FUNCTION
// ============================================================================

async function main() {
  const query = process.argv[2];

  if (!query) {
    console.log('\n❌ No search query provided.\n');
    console.log('Usage:');
    console.log('  yarn start:hybrid "your query"\n');
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
    console.log('Debug Commands:');
    console.log('  yarn start:hybrid list-relations  (shows all relationships in database)\n');
    console.log('Note: Database is reset and recreated on every run for consistent results.\n');
    return;
  }

  // Always reset and recreate database for consistent results
  console.log('🔄 Resetting database for fresh setup...\n');
  await resetDatabase();
  await setupDatabase();
  console.log();

  // Run multi-model search
  await search(query);
}

main().catch((e: any) => {
  console.error('\n❌ An unhandled error occurred:');
  console.error(e);
  process.exit(1);
});

