import { EventEmitter } from 'events';
import { pipeline } from '@xenova/transformers';
import hnswlib from 'hnswlib-node';
import axios from 'axios';
import * as cheerio from 'cheerio';
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  Tool,
  Resource,
  McpError,
  ErrorCode,
  ListToolsRequestSchema,
  ListResourcesRequestSchema,
  CallToolRequestSchema,
  ReadResourceRequestSchema
} from '@modelcontextprotocol/sdk/types.js';

interface DocItem {
  title: string;
  content: string;
  url: string;
  type: 'doc' | 'example';
}

const TOOLS: Tool[] = [
  {
    name: 'search_metal_docs',
    description: 'Search Metal Framework documentation and code examples using natural language queries',
    inputSchema: {
      type: 'object',
      properties: {
        query: {
          type: 'string',
          description: 'Natural language query about Metal Framework',
        },
        limit: {
          type: 'number',
          description: 'Maximum number of results to return',
          default: 3,
        },
      },
      required: ['query'],
    },
  },
  {
    name: 'generate_metal_code',
    description: 'Generate Metal Framework code for common tasks',
    inputSchema: {
      type: 'object',
      properties: {
        task: {
          type: 'string',
          description: 'Description of the Metal task to generate code for',
        },
        language: {
          type: 'string',
          description: 'Programming language (objective-c, swift, or metal)',
          default: 'swift',
        },
      },
      required: ['task'],
    },
  },
];

const RESOURCES: Resource[] = [
  {
    uri: 'metal://docs/getting-started',
    name: 'Metal Getting Started Guide',
    description: 'Comprehensive guide for getting started with Metal Framework',
  },
  {
    uri: 'metal://docs/best-practices',
    name: 'Metal Best Practices',
    description: 'Best practices and optimization tips for Metal Framework',
  },
];

export class MetalExpertServer {
  private docs: DocItem[] = [];
  private vectorStore: any;
  private embedder: any = null;
  private initialized = false;
  private server: Server;

  constructor() {
    this.vectorStore = new hnswlib.HierarchicalNSW('cosine', 384);
    this.server = new Server(
      {
        name: 'metal-mcp-server',
        version: '0.1.0',
      },
      {
        capabilities: {
          tools: TOOLS.reduce((acc, tool) => {
            acc[tool.name] = tool;
            return acc;
          }, {} as Record<string, Tool>),
          resources: RESOURCES.reduce((acc, resource) => {
            acc[resource.uri] = resource;
            return acc;
          }, {} as Record<string, Resource>)
        }
      }
    );
    this.setupHandlers();
  }

  private setupHandlers() {
    this.server.setRequestHandler(ListToolsRequestSchema, async () => ({
      tools: TOOLS
    }));

    this.server.setRequestHandler(ListResourcesRequestSchema, async () => ({
      resources: RESOURCES
    }));

    this.server.setRequestHandler(CallToolRequestSchema, this.handleToolRequest.bind(this));
    this.server.setRequestHandler(ReadResourceRequestSchema, this.handleResourceRequest.bind(this));
  }

  async run() {
    const transport = new StdioServerTransport();
    await this.server.connect(transport);
    console.error('Metal MCP server running on stdio');
  }

  private async initEmbedder() {
    if (!this.embedder) {
      this.embedder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
    }
    return this.embedder;
  }

  private async initializeRAG() {
    if (this.initialized) return;

    // Fetch Metal documentation
    const response = await axios.get('https://developer.apple.com/documentation/metal');
    const $ = cheerio.load(response.data);
    
    // Extract documentation content
    $('.documentation-content').each((i, elem) => {
      const title = $(elem).find('h1').text();
      const content = $(elem).text();
      const url = $(elem).find('a').attr('href') || '';
      
      this.docs.push({
        title,
        content,
        url,
        type: 'doc'
      });
    });

    // Add code examples
    this.docs.push({
      title: 'Basic Metal Setup',
      content: `
      // Create a Metal device
      id<MTLDevice> device = MTLCreateSystemDefaultDevice();
      
      // Create a command queue
      id<MTLCommandQueue> commandQueue = [device newCommandQueue];
      
      // Create a Metal library
      id<MTLLibrary> defaultLibrary = [device newDefaultLibrary];
      `,
      url: 'examples/basic_setup',
      type: 'example'
    });

    // Initialize vector store
    this.vectorStore.initIndex(this.docs.length, true);

    // Get embeddings and add to vector store
    const embedder = await this.initEmbedder();
    for (let i = 0; i < this.docs.length; i++) {
      const embedding = await embedder(this.docs[i].content, { pooling: 'mean', normalize: true });
      this.vectorStore.addPoint(embedding.data, i);
    }

    this.initialized = true;
  }

  async handleToolRequest(request: { params: { name: string; arguments?: Record<string, unknown> } }): Promise<any> {
    await this.initializeRAG();

    switch (request.params.name) {
      case 'search_metal_docs': {
        const query = request.params.arguments?.query as string;
        const limit = (request.params.arguments?.limit as number) ?? 3;
        
        // Get query embedding
        const embedder = await this.initEmbedder();
        const queryEmbedding = await embedder(query, { pooling: 'mean', normalize: true });
        
        // Search vector store
        const results = this.vectorStore.searchKnn(queryEmbedding.data, limit);
        
        // Format results
        const searchResults = results.neighbors.map((idx: number) => {
          const doc = this.docs[idx];
          return {
            title: doc.title,
            content: doc.content,
            url: doc.url,
            type: doc.type,
          };
        });

        return {
          content: [
            {
              type: 'text',
              text: JSON.stringify(searchResults, null, 2),
            },
          ],
        };
      }

      case 'generate_metal_code': {
        const task = request.params.arguments?.task as string;
        const language = (request.params.arguments?.language as string) ?? 'swift';

        // Search for relevant examples
        const embedder = await this.initEmbedder();
        const taskEmbedding = await embedder(task, { pooling: 'mean', normalize: true });
        const results = this.vectorStore.searchKnn(taskEmbedding.data, 3);

        // Use examples to generate contextual code
        const examples = results.neighbors.map((idx: number) => this.docs[idx]);
        
        // Generate code based on task and examples
        let code = '';
        if (task.includes('compute')) {
          code = this.generateComputeCode(language);
        } else if (task.includes('render')) {
          code = this.generateRenderCode(language);
        } else {
          code = this.generateBasicCode(language);
        }

        return {
          content: [
            {
              type: 'text',
              text: code,
            },
          ],
        };
      }

      default:
        throw new McpError(
          ErrorCode.MethodNotFound,
          `Unknown tool: ${request.params.name}`
        );
    }
  }

  async handleResourceRequest(request: { params: { uri: string } }): Promise<any> {
    await this.initializeRAG();

    const uri = request.params.uri;
    let content = '';

    switch (uri) {
      case 'metal://docs/getting-started':
        content = this.docs.filter(d => d.type === 'doc' && d.title.includes('Getting Started'))
          .map(d => d.content)
          .join('\n\n');
        break;

      case 'metal://docs/best-practices':
        content = this.docs.filter(d => d.type === 'doc' && d.title.includes('Best Practices'))
          .map(d => d.content)
          .join('\n\n');
        break;

      default:
        throw new McpError(
          ErrorCode.InvalidRequest,
          `Unknown resource URI: ${uri}`
        );
    }

    return {
      contents: [
        {
          uri,
          mimeType: 'text/plain',
          text: content,
        },
      ],
    };
  }

  async listTools(): Promise<Tool[]> {
    return TOOLS;
  }

  async listResources(): Promise<Resource[]> {
    return RESOURCES;
  }

  private generateComputeCode(language: string): string {
    switch (language) {
      case 'swift':
        return `
import Metal

guard let device = MTLCreateSystemDefaultDevice() else {
    fatalError("GPU not available")
}

let commandQueue = device.makeCommandQueue()!

// Create compute pipeline
let library = device.makeDefaultLibrary()!
let function = library.makeFunction(name: "compute_function")!
let pipeline = try! device.makeComputePipelineState(function: function)

// Create command buffer and encoder
let commandBuffer = commandQueue.makeCommandBuffer()!
let computeEncoder = commandBuffer.makeComputeCommandEncoder()!

computeEncoder.setComputePipelineState(pipeline)
// Set buffer bindings and dispatch here

computeEncoder.endEncoding()
commandBuffer.commit()
`;

      case 'objective-c':
        return `
id<MTLDevice> device = MTLCreateSystemDefaultDevice();
id<MTLCommandQueue> commandQueue = [device newCommandQueue];

// Create compute pipeline
id<MTLLibrary> library = [device newDefaultLibrary];
id<MTLFunction> function = [library newFunctionWithName:@"compute_function"];
id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:nil];

// Create command buffer and encoder
id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];

[computeEncoder setComputePipelineState:pipeline];
// Set buffer bindings and dispatch here

[computeEncoder endEncoding];
[commandBuffer commit];
`;

      case 'metal':
        return `
#include <metal_stdlib>
using namespace metal;

kernel void compute_function(
    device float *input_buffer [[buffer(0)]],
    device float *output_buffer [[buffer(1)]],
    uint index [[thread_position_in_grid]])
{
    // Compute work here
    output_buffer[index] = input_buffer[index] * 2.0;
}
`;

      default:
        throw new Error(`Unsupported language: ${language}`);
    }
  }

  private generateRenderCode(language: string): string {
    switch (language) {
      case 'swift':
        return `
import Metal
import MetalKit

guard let device = MTLCreateSystemDefaultDevice() else {
    fatalError("GPU not available")
}

let commandQueue = device.makeCommandQueue()!

// Create render pipeline
let library = device.makeDefaultLibrary()!
let vertexFunction = library.makeFunction(name: "vertex_function")!
let fragmentFunction = library.makeFunction(name: "fragment_function")!

let pipelineDescriptor = MTLRenderPipelineDescriptor()
pipelineDescriptor.vertexFunction = vertexFunction
pipelineDescriptor.fragmentFunction = fragmentFunction
pipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm

let pipeline = try! device.makeRenderPipelineState(descriptor: pipelineDescriptor)

// Create command buffer and encoder
let commandBuffer = commandQueue.makeCommandBuffer()!
let renderPassDescriptor = MTLRenderPassDescriptor()
let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor)!

renderEncoder.setRenderPipelineState(pipeline)
// Set vertex buffers and draw here

renderEncoder.endEncoding()
commandBuffer.commit()
`;

      case 'objective-c':
        return `
id<MTLDevice> device = MTLCreateSystemDefaultDevice();
id<MTLCommandQueue> commandQueue = [device newCommandQueue];

// Create render pipeline
id<MTLLibrary> library = [device newDefaultLibrary];
id<MTLFunction> vertexFunction = [library newFunctionWithName:@"vertex_function"];
id<MTLFunction> fragmentFunction = [library newFunctionWithName:@"fragment_function"];

MTLRenderPipelineDescriptor *pipelineDescriptor = [[MTLRenderPipelineDescriptor alloc] init];
pipelineDescriptor.vertexFunction = vertexFunction;
pipelineDescriptor.fragmentFunction = fragmentFunction;
pipelineDescriptor.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;

id<MTLRenderPipelineState> pipeline = [device newRenderPipelineStateWithDescriptor:pipelineDescriptor error:nil];

// Create command buffer and encoder
id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
MTLRenderPassDescriptor *renderPassDescriptor = [MTLRenderPassDescriptor renderPassDescriptor];
id<MTLRenderCommandEncoder> renderEncoder = [commandBuffer renderCommandEncoderWithDescriptor:renderPassDescriptor];

[renderEncoder setRenderPipelineState:pipeline];
// Set vertex buffers and draw here

[renderEncoder endEncoding];
[commandBuffer commit];
`;

      case 'metal':
        return `
#include <metal_stdlib>
using namespace metal;

struct VertexIn {
    float3 position [[attribute(0)]];
    float2 texCoord [[attribute(1)]];
};

struct VertexOut {
    float4 position [[position]];
    float2 texCoord;
};

vertex VertexOut vertex_function(
    VertexIn in [[stage_in]])
{
    VertexOut out;
    out.position = float4(in.position, 1.0);
    out.texCoord = in.texCoord;
    return out;
}

fragment float4 fragment_function(
    VertexOut in [[stage_in]],
    texture2d<float> texture [[texture(0)]],
    sampler textureSampler [[sampler(0)]])
{
    return texture.sample(textureSampler, in.texCoord);
}
`;

      default:
        throw new Error(`Unsupported language: ${language}`);
    }
  }

  private generateBasicCode(language: string): string {
    switch (language) {
      case 'swift':
        return `
import Metal

guard let device = MTLCreateSystemDefaultDevice() else {
    fatalError("GPU not available")
}

let commandQueue = device.makeCommandQueue()!
let library = device.makeDefaultLibrary()!

// Create buffers
let buffer = device.makeBuffer(length: 1024, options: .storageModeShared)!

// Create command buffer
let commandBuffer = commandQueue.makeCommandBuffer()!
commandBuffer.commit()
`;

      case 'objective-c':
        return `
id<MTLDevice> device = MTLCreateSystemDefaultDevice();
id<MTLCommandQueue> commandQueue = [device newCommandQueue];
id<MTLLibrary> library = [device newDefaultLibrary];

// Create buffers
id<MTLBuffer> buffer = [device newBufferWithLength:1024 options:MTLResourceStorageModeShared];

// Create command buffer
id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
[commandBuffer commit];
`;

      case 'metal':
        return `
#include <metal_stdlib>
using namespace metal;

struct Uniforms {
    float4x4 modelMatrix;
    float4x4 viewMatrix;
    float4x4 projectionMatrix;
};
`;

      default:
        throw new Error(`Unsupported language: ${language}`);
    }
  }
}
