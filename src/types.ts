export interface Server {
  name: string;
  version: string;
}

export interface ServerCapabilities {
  resources: Record<string, unknown>;
  tools: Record<string, unknown>;
}

export interface ToolRequest {
  params: {
    name: string;
    arguments: Record<string, any>;
  };
}

export interface ResourceRequest {
  params: {
    uri: string;
  };
}

export interface Content {
  type: string;
  text: string;
}

export interface ToolResponse {
  content: Content[];
}

export interface ResourceResponse {
  contents: {
    uri: string;
    mimeType: string;
    text: string;
  }[];
}

export interface Tool {
  name: string;
  description: string;
  inputSchema: {
    type: string;
    properties: Record<string, any>;
    required?: string[];
  };
}

export interface Resource {
  uri: string;
  name: string;
  description?: string;
  mimeType?: string;
}

export interface ErrorResponse {
  code: number;
  message: string;
}

export class McpError extends Error {
  constructor(public code: number, message: string) {
    super(message);
    this.name = 'McpError';
  }
}

export const ErrorCode = {
  InvalidRequest: 400,
  MethodNotFound: 404,
  InternalError: 500,
} as const;
