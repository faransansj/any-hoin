type PostCSSConfig = {
  plugins: Record<string, Record<string, never>>;
};

const config = {
  plugins: { tailwindcss: {}, autoprefixer: {} },
} satisfies PostCSSConfig;

export = config;
