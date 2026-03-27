import { Module } from "vuex";
import { State } from "../index";
import * as tbService from "@/services/tensorbasicService";

export interface TensorBasicState {
  files: string[];
  currentFile: string;
  sourceText: string;
  highlightHtml: string;
  mermaidSrc: string;
  umapData: any[];
  activeTab: string;
}

const tensorbasicModule: Module<TensorBasicState, State> = {
  namespaced: true,
  state: () => ({
    files: [],
    currentFile: "",
    sourceText: "",
    highlightHtml: "",
    mermaidSrc: "",
    umapData: [],
    activeTab: "source",
  }),
  mutations: {
    setFiles(state, files: string[]) {
      state.files = files;
    },
    setCurrentFile(state, file: string) {
      state.currentFile = file;
    },
    setSourceText(state, source: string) {
      state.sourceText = source;
    },
    setHighlightHtml(state, html: string) {
      state.highlightHtml = html;
    },
    setMermaidSrc(state, src: string) {
      state.mermaidSrc = src;
    },
    setUmapData(state, data: any[]) {
      state.umapData = data;
    },
    setActiveTab(state, tab: string) {
      state.activeTab = tab;
    },
  },
  actions: {
    async loadFiles({ commit }) {
      try {
        const files = await tbService.getFiles();
        commit("setFiles", files);
      } catch (e) {
        console.error("Failed to load tensorBASIC files:", e);
      }
    },
    async loadFile({ commit }, file: string) {
      commit("setCurrentFile", file);
      try {
        const [source, highlight, mermaid, umap] = await Promise.all([
          tbService.getSource(file),
          tbService.getHighlight(file),
          tbService.getMermaid(file),
          tbService.getUmap(file),
        ]);
        commit("setSourceText", source);
        commit("setHighlightHtml", highlight);
        commit("setMermaidSrc", mermaid);
        commit("setUmapData", umap);
      } catch (e) {
        console.error("Failed to load file:", file, e);
      }
    },
    async refreshFile({ state, dispatch }) {
      if (state.currentFile) {
        await dispatch("loadFile", state.currentFile);
      }
    },
  },
};

export default tensorbasicModule;
