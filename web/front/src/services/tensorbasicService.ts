import axios from "axios";

const tbServerUrl = "http://localhost:8080";

export async function getFiles(): Promise<string[]> {
  const response = await axios.get(`${tbServerUrl}/api/files`);
  return response.data;
}

export async function getSource(file: string): Promise<string> {
  const response = await axios.get(`${tbServerUrl}/api/source`, {
    params: { file },
  });
  return response.data;
}

export async function getHighlight(file: string): Promise<string> {
  const response = await axios.get(`${tbServerUrl}/api/highlight`, {
    params: { file },
  });
  return response.data;
}

export async function getMermaid(file: string): Promise<string> {
  const response = await axios.get(`${tbServerUrl}/api/mermaid`, {
    params: { file },
  });
  return response.data;
}

export async function getUmap(file: string): Promise<any[]> {
  const response = await axios.get(`${tbServerUrl}/api/umap`, {
    params: { file },
  });
  return response.data;
}

export async function getAst(file: string): Promise<string> {
  const response = await axios.get(`${tbServerUrl}/api/ast`, {
    params: { file },
  });
  return response.data;
}
